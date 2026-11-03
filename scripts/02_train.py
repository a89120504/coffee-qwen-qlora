# -*- coding: utf-8 -*-
"""
02_train.py
使用 Unsloth + QLoRA 在 RTX 3070 (8GB) 上微調 Qwen2.5-3B-Instruct
Unsloth 可加速訓練約 1.5-2x 並降低 VRAM 用量約 30-40%
設定 USE_UNSLOTH = False 可切回原生 HuggingFace 流程
"""

import json
from pathlib import Path

# ========== 切換開關 ==========
USE_UNSLOTH = False  # Windows 上 Unsloth Triton kernel 不穩定，使用原生 HuggingFace

# Unsloth 必須在 trl/transformers 之前 import，才能正確套用優化
if USE_UNSLOTH:
    from unsloth import FastLanguageModel

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
import torch

if not USE_UNSLOTH:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ========== 可調參數 ==========
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# Unsloth 提供預優化版本，載入更快
UNSLOTH_MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH = 512

# LoRA 參數
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "v_proj",
]

# 訓練參數
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4  # 等效 batch = 8
NUM_EPOCHS = 2
WARMUP_STEPS = 10
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
SAVE_STEPS = 200


def format_chat(example, tokenizer):
    """將 conversations 格式轉成 tokenizer 的 chat template"""
    text = tokenizer.apply_chat_template(
        example["conversations"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def load_model_unsloth():
    """使用 Unsloth 載入模型（更快、更省 VRAM）"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=UNSLOTH_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # 自動偵測 float16/bfloat16
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth 優化的 gradient checkpointing
    )
    return model, tokenizer


def load_model_hf():
    """使用原生 HuggingFace 載入模型（fallback）"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def main():
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "processed" / "coffee_train.json"
    output_dir = base_dir / "models" / "coffee-qwen2.5-3b-lora"

    backend = "Unsloth" if USE_UNSLOTH else "HuggingFace"
    print("=" * 60)
    print(f"  訓練後端：{backend}")
    print("=" * 60)

    # 1. 載入訓練資料與驗證資料
    print("[1/5] 載入訓練與驗證資料...")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"  訓練資料：{len(raw_data)} 筆")
    dataset = Dataset.from_list(raw_data)

    val_path = base_dir / "data" / "processed" / "coffee_val.json"
    eval_dataset = None
    if val_path.exists():
        with open(val_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        print(f"  驗證資料：{len(val_data)} 筆")
        eval_dataset = Dataset.from_list(val_data)
    else:
        print("  [WARNING] 未找到 coffee_val.json，將不監控驗證 loss")

    # 2-4. 載入模型 + 量化 + LoRA
    print(f"[2-4/5] 載入模型 + QLoRA（{backend}）...")
    if USE_UNSLOTH:
        model, tokenizer = load_model_unsloth()
    else:
        model, tokenizer = load_model_hf()
    model.print_trainable_parameters()

    # 格式化資料
    print("  格式化訓練資料...")
    dataset = dataset.map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            lambda x: format_chat(x, tokenizer),
            remove_columns=eval_dataset.column_names,
        )

    # 5. 訓練
    gpu_stats = torch.cuda.get_device_properties(0)
    print(f"\n  GPU: {gpu_stats.name}")
    print(f"  VRAM: {gpu_stats.total_memory / 1024**3:.1f} GB")
    print(f"[5/5] 開始訓練...")
    print("=" * 60)

    # 根據有無 validation set 設定評估策略
    eval_args = {}
    if eval_dataset is not None:
        eval_args = {
            "eval_strategy": "steps",
            "eval_steps": SAVE_STEPS,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        }

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        gradient_checkpointing=not USE_UNSLOTH,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        # SFTConfig 專屬參數（trl 0.24+）
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=True,
        eos_token=tokenizer.eos_token,
        **eval_args,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    trainer.train()

    # 儲存
    print("\n" + "=" * 60)
    print("[完成] 儲存 LoRA adapter...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[儲存] {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

