# -*- coding: utf-8 -*-
"""
04_ab_comparison.py
A/B 對比：微調前 vs 微調後的咖啡品鑑助手
生成 Markdown 格式的對比報告，供簡報使用
"""

from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

SYSTEM_PROMPT = (
    "你是一位專業的咖啡品鑑師與顧問，擁有豐富的精品咖啡知識。"
    "你熟悉世界各產區的咖啡特色、不同烘焙程度的風味差異、以及各種沖煮方法。"
    "請用專業但易懂的方式，以繁體中文回答使用者關於咖啡的各種問題，"
    "包括風味描述、咖啡推薦、沖煮建議等。"
)

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

TEST_QUESTIONS = [
    "推薦一款帶有花香和莓果風味的淺焙咖啡",
    "衣索比亞和哥倫比亞的咖啡有什麼不同？",
    "我喜歡巧克力和堅果風味，不想太酸，推薦什麼咖啡？",
    "中深焙咖啡應該怎麼沖泡比較好喝？",
    "什麼樣的咖啡能拿到 95 分以上的評分？",
    "新手想嘗試精品咖啡，應該從哪裡開始？",
    "日曬和水洗處理法的咖啡風味差異是什麼？",
]


def generate_response(model, tokenizer, question):
    """用模型生成回答"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def main():
    base_dir = Path(__file__).parent.parent
    lora_dir = base_dir / "models" / "coffee-qwen2.5-3b-lora"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # === 載入原始模型（微調前）===
    print("[1/4] 載入原始模型（微調前）...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    base_model.eval()

    # 生成「微調前」回答
    print("[2/4] 生成微調前回答...")
    before_answers = []
    for i, q in enumerate(TEST_QUESTIONS):
        print(f"  問題 {i+1}/{len(TEST_QUESTIONS)}: {q[:30]}...")
        ans = generate_response(base_model, tokenizer, q)
        before_answers.append(ans)

    # 載入 LoRA（在 base_model 上疊加）
    print("[3/4] 載入微調後 LoRA adapter...")
    ft_model = PeftModel.from_pretrained(base_model, str(lora_dir))
    ft_model.eval()

    # 生成「微調後」回答
    print("[4/4] 生成微調後回答...")
    after_answers = []
    for i, q in enumerate(TEST_QUESTIONS):
        print(f"  問題 {i+1}/{len(TEST_QUESTIONS)}: {q[:30]}...")
        ans = generate_response(ft_model, tokenizer, q)
        after_answers.append(ans)

    # === 生成對比報告 ===
    report_lines = [
        "# ☕ A/B 對比報告：咖啡品鑑助手",
        f"\n> 生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> Base Model: Qwen2.5-3B-Instruct",
        f"> Fine-tuned: QLoRA (rank=16, alpha=32, 3 epochs)\n",
        "---\n",
    ]

    for i, q in enumerate(TEST_QUESTIONS):
        report_lines.append(f"## 問題 {i+1}: {q}\n")
        report_lines.append("### 🔵 微調前（原始模型）\n")
        report_lines.append(f"{before_answers[i]}\n")
        report_lines.append("### 🟢 微調後（咖啡品鑑助手）\n")
        report_lines.append(f"{after_answers[i]}\n")
        report_lines.append("---\n")

    report_path = results_dir / "ab_comparison.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n{'='*60}")
    print(f"[完成] A/B 對比報告已儲存至: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
