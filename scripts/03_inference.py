# -*- coding: utf-8 -*-
"""
03_inference.py
載入微調後的咖啡品鑑助手模型，進行互動式問答
"""

from pathlib import Path
import re
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


def load_model(lora_path):
    """載入 base model + LoRA adapter"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, user_input):
    """生成回答"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
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


def post_process(response, user_input):
    """推論後處理：攜截明顮的數值與邏輯錯誤"""
    warnings = []

    # 1. 檢查評分門檻一致性
    threshold_match = re.search(r'(\d{2,3})\s*分以上', user_input)
    if threshold_match:
        threshold = int(threshold_match.group(1))
        scores = [int(s) for s in re.findall(r'(\d{2,3})/100', response)]
        bad_scores = [s for s in scores if s < threshold]
        if bad_scores:
            warnings.append(
                f"⚠️ 注意：上述推薦中部分咖啡評分（{', '.join(str(s) for s in bad_scores)} 分）"
                f"未達 {threshold} 分門檻，建議進一步確認。"
            )

    # 2. 檢查烘焙度與沖煮建議一致性
    roast_temp_rules = {
        "深焙": {"max_temp": 90, "label": "85-88°C"},
        "中深焙": {"max_temp": 91, "label": "86-90°C"},
        "淺焙": {"min_temp": 89, "label": "90-93°C"},
    }
    for roast, rule in roast_temp_rules.items():
        if roast in response:
            temps = re.findall(r'(\d{2,3})°C', response)
            for t in temps:
                t_val = int(t)
                if "max_temp" in rule and t_val > rule["max_temp"]:
                    warnings.append(
                        f"⚠️ 提醒：{roast}咖啡建議水溫為 {rule['label']}，"
                        f"上述建議的 {t_val}°C 可能偏高。"
                    )
                    break
                if "min_temp" in rule and t_val < rule["min_temp"]:
                    warnings.append(
                        f"⚠️ 提醒：{roast}咖啡建議水溫為 {rule['label']}，"
                        f"上述建議的 {t_val}°C 可能偏低。"
                    )
                    break
            break  # 只檢查第一個匹配的烘焙度

    if warnings:
        response += "\n\n" + "\n".join(warnings)

    return response


def main():
    base_dir = Path(__file__).parent.parent
    lora_path = base_dir / "models" / "coffee-qwen2.5-3b-lora"

    print("[載入] 微調後的咖啡品鑑助手模型...")
    model, tokenizer = load_model(str(lora_path))
    print("[就緒] 輸入問題開始對話（輸入 'quit' 結束）\n")

    while True:
        user_input = input("☕ 你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("再見！享受你的咖啡 ☕")
            break
        if not user_input:
            continue
        response = generate(model, tokenizer, user_input)
        response = post_process(response, user_input)
        print(f"\n🤖 品鑑師: {response}\n")


if __name__ == "__main__":
    main()
