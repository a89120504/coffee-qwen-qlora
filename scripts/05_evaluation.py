# -*- coding: utf-8 -*-
"""
05_evaluation.py
自動化評估腳本：用 test set 量化模型的準確度與一致性
支援規則式評分，不需要 GPT 當裁判
"""

# === Workaround: Windows NTFS TxF bug (WinError 6714) ===
# pyarrow 的 C 擴充模組載入 DLL 時會改變執行緒的 NTFS 交易上下文，
# 導致 Python import 機制中 _fill_cache (os.listdir) 呼叫失敗。
# 策略一：patch importlib 的 FileFinder._fill_cache 讓它容錯
# 策略二：在 sys.modules 中預先 mock pyarrow，阻止真正的 C 擴充載入
import sys as _sys
import types as _types

# 策略一：Patch FileFinder._fill_cache
try:
    import importlib._bootstrap_external as _be
    _orig_fill_cache = _be.FileFinder._fill_cache

    def _patched_fill_cache(self):
        try:
            return _orig_fill_cache(self)
        except OSError as e:
            if hasattr(e, 'winerror') and e.winerror == 6714:
                self._path_cache = set()
                if hasattr(self, '_relaxed_path_cache'):
                    self._relaxed_path_cache = set()
                return
            raise

    _be.FileFinder._fill_cache = _patched_fill_cache
except Exception:
    pass

# 策略二：Mock pyarrow（本腳本不需要 pyarrow/pandas/sklearn）
_mock_pa = _types.ModuleType('pyarrow')
_mock_pa.__version__ = '24.0.0'
_mock_pa.__path__ = []
_mock_pa.lib = _types.ModuleType('pyarrow.lib')
_sys.modules['pyarrow'] = _mock_pa
_sys.modules['pyarrow.lib'] = _mock_pa.lib
# === End Workaround ===

import json
import re
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

# ========== 評估用測試題（含自動化檢查規則） ==========

EVAL_QUESTIONS = [
    # --- 數值準確度測試 ---
    {
        "category": "數值準確度",
        "question": "推薦一款 95 分以上的咖啡。",
        "checks": [
            ("has_score", "回答中應包含評分"),
            ("score_above_threshold", 95, "推薦的評分應 ≥ 95"),
        ],
    },
    {
        "category": "數值準確度",
        "question": "有沒有 90 分以上的咖啡？",
        "checks": [
            ("has_score", "回答中應包含評分"),
            ("score_above_threshold", 90, "推薦的評分應 ≥ 90"),
        ],
    },
    {
        "category": "數值準確度",
        "question": "推薦一款 93 分以上的咖啡。",
        "checks": [
            ("has_score", "回答中應包含評分"),
            ("score_above_threshold", 93, "推薦的評分應 ≥ 93"),
        ],
    },
    # --- 烘焙度一致性測試 ---
    {
        "category": "烘焙一致性",
        "question": "推薦一款深焙咖啡，並告訴我怎麼沖。",
        "checks": [
            ("contains_keyword", "深焙", "回答應提及深焙"),
            ("temp_max", 90, "深焙水溫不應超過 90°C"),
        ],
    },
    {
        "category": "烘焙一致性",
        "question": "推薦一款淺焙咖啡，並告訴我怎麼沖。",
        "checks": [
            ("contains_keyword", "淺焙", "回答應提及淺焙"),
            ("temp_min", 89, "淺焙水溫不應低於 89°C"),
        ],
    },
    {
        "category": "烘焙一致性",
        "question": "中深焙咖啡應該怎麼沖比較好喝？",
        "checks": [
            ("contains_keyword", "中深焙", "回答應提及中深焙"),
            ("temp_max", 91, "中深焙水溫不應超過 91°C"),
        ],
    },
    # --- 格式完整度測試 ---
    {
        "category": "格式完整度",
        "question": "請介紹一款衣索比亞的咖啡。",
        "checks": [
            ("contains_keyword", "衣索比亞", "應提及產區"),
            ("has_brew_tip", "應包含沖煮建議"),
            ("has_score", "應包含評分"),
        ],
    },
    {
        "category": "格式完整度",
        "question": "我喜歡巧克力和堅果風味的咖啡，有推薦嗎？",
        "checks": [
            ("contains_any", ["巧克力", "堅果", "可可"], "應提及相關風味"),
            ("has_brew_tip", "應包含沖煮建議"),
        ],
    },
    # --- 產區知識測試 ---
    {
        "category": "產區知識",
        "question": "衣索比亞和哥倫比亞的咖啡有什麼不同？",
        "checks": [
            ("contains_keyword", "衣索比亞", "應提及衣索比亞"),
            ("contains_keyword", "哥倫比亞", "應提及哥倫比亞"),
            ("min_length", 100, "比較回答應有足夠長度"),
        ],
    },
    {
        "category": "產區知識",
        "question": "日曬和水洗處理法的咖啡風味差異是什麼？",
        "checks": [
            ("contains_keyword", "日曬", "應提及日曬"),
            ("contains_keyword", "水洗", "應提及水洗"),
            ("min_length", 80, "比較回答應有足夠長度"),
        ],
    },
]


# ========== 檢查函式 ==========

def check_has_score(response):
    """檢查回答中是否包含評分"""
    return bool(re.search(r'\d{2,3}/100', response))


def check_score_above_threshold(response, threshold):
    """檢查回答中的評分是否都 ≥ 門檻"""
    scores = [int(s) for s in re.findall(r'(\d{2,3})/100', response)]
    if not scores:
        return False
    return all(s >= threshold for s in scores)


def check_contains_keyword(response, keyword):
    """檢查回答是否包含關鍵字"""
    return keyword in response


def check_contains_any(response, keywords):
    """檢查回答是否包含任一關鍵字"""
    return any(k in response for k in keywords)


def check_has_brew_tip(response):
    """檢查回答是否包含沖煮建議"""
    brew_keywords = ["沖煮", "水溫", "研磨", "粉水比", "手沖", "濾壓", "義式"]
    return any(k in response for k in brew_keywords)


def check_temp_max(response, max_temp):
    """檢查回答中的水溫是否不超過上限"""
    temps = [int(t) for t in re.findall(r'(\d{2,3})°C', response)]
    if not temps:
        return True  # 沒提水溫就不算錯
    return all(t <= max_temp for t in temps)


def check_temp_min(response, min_temp):
    """檢查回答中的水溫是否不低於下限"""
    temps = [int(t) for t in re.findall(r'(\d{2,3})°C', response)]
    if not temps:
        return True
    return all(t >= min_temp for t in temps)


def check_min_length(response, min_len):
    """檢查回答長度是否足夠"""
    return len(response) >= min_len


def run_checks(response, checks):
    """執行一組檢查，回傳 (通過數, 總數, 詳細結果)"""
    results = []
    for check in checks:
        check_type = check[0]
        if check_type == "has_score":
            passed = check_has_score(response)
            desc = check[1]
        elif check_type == "score_above_threshold":
            passed = check_score_above_threshold(response, check[1])
            desc = check[2]
        elif check_type == "contains_keyword":
            passed = check_contains_keyword(response, check[1])
            desc = check[2]
        elif check_type == "contains_any":
            passed = check_contains_any(response, check[1])
            desc = check[2]
        elif check_type == "has_brew_tip":
            passed = check_has_brew_tip(response)
            desc = check[1]
        elif check_type == "temp_max":
            passed = check_temp_max(response, check[1])
            desc = check[2]
        elif check_type == "temp_min":
            passed = check_temp_min(response, check[1])
            desc = check[2]
        elif check_type == "min_length":
            passed = check_min_length(response, check[1])
            desc = check[2]
        else:
            passed = False
            desc = f"未知檢查類型: {check_type}"

        results.append({"desc": desc, "passed": passed})

    passed_count = sum(1 for r in results if r["passed"])
    return passed_count, len(results), results


# ========== 主程式 ==========

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

    # 載入模型
    print("=" * 60)
    print("[1/3] 載入微調後模型...")
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
    model = PeftModel.from_pretrained(model, str(lora_dir))
    model.eval()

    # 執行評估
    print("[2/3] 執行評估問題...")
    all_results = []
    total_passed = 0
    total_checks = 0

    for i, item in enumerate(EVAL_QUESTIONS):
        q = item["question"]
        cat = item["category"]
        print(f"  [{i+1}/{len(EVAL_QUESTIONS)}] ({cat}) {q[:40]}...")

        response = generate_response(model, tokenizer, q)
        passed, total, details = run_checks(response, item["checks"])

        total_passed += passed
        total_checks += total

        result = {
            "category": cat,
            "question": q,
            "response": response,
            "passed": passed,
            "total": total,
            "details": details,
        }
        all_results.append(result)
        status = "✅" if passed == total else "⚠️"
        print(f"       {status} {passed}/{total} 通過")

    # 生成報告
    print("[3/3] 生成評估報告...")
    accuracy = total_passed / total_checks * 100 if total_checks > 0 else 0

    # 按類別統計
    cat_stats = {}
    for r in all_results:
        cat = r["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"passed": 0, "total": 0}
        cat_stats[cat]["passed"] += r["passed"]
        cat_stats[cat]["total"] += r["total"]

    report_lines = [
        "# ☕ 咖啡品鑑助手 — 自動評估報告",
        "",
        f"> 生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> Base Model: {BASE_MODEL}",
        f"> Fine-tuned: QLoRA adapter",
        "",
        "---",
        "",
        "## 總覽",
        "",
        f"**整體通過率：{accuracy:.1f}%**（{total_passed}/{total_checks}）",
        "",
        "| 評估類別 | 通過 | 總數 | 通過率 |",
        "|----------|------|------|--------|",
    ]
    for cat, stats in cat_stats.items():
        rate = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        report_lines.append(f"| {cat} | {stats['passed']} | {stats['total']} | {rate:.0f}% |")

    report_lines.extend(["", "---", ""])

    for i, r in enumerate(all_results):
        status = "✅" if r["passed"] == r["total"] else "⚠️"
        report_lines.append(f"## {status} 問題 {i+1}（{r['category']}）：{r['question']}")
        report_lines.append("")
        report_lines.append("### 模型回答")
        report_lines.append("")
        report_lines.append(r["response"])
        report_lines.append("")
        report_lines.append("### 檢查結果")
        report_lines.append("")
        for d in r["details"]:
            icon = "✅" if d["passed"] else "❌"
            report_lines.append(f"- {icon} {d['desc']}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

    report_path = results_dir / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # 同時存 JSON 方便後續分析
    json_path = results_dir / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy,
            "total_passed": total_passed,
            "total_checks": total_checks,
            "category_stats": cat_stats,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"[完成] 評估結果")
    print(f"  整體通過率：{accuracy:.1f}%（{total_passed}/{total_checks}）")
    for cat, stats in cat_stats.items():
        rate = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat}：{rate:.0f}%（{stats['passed']}/{stats['total']}）")
    print(f"\n  報告：{report_path}")
    print(f"  JSON：{json_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
