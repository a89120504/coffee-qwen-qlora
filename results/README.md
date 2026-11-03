# 既有成果索引

這些報告是專案原有的歷史成果，整理時未重新生成或更改內容。

| 檔案 | 內容與檔內日期 |
| --- | --- |
| [ab_comparison.md](ab_comparison.md) | 簡報精選版 A/B，2026-05-30 |
| [ab_comparison_full.md](ab_comparison_full.md) | 完整逐字 A/B，2026-06-05 |
| [evaluation_report.md](evaluation_report.md) | 規則式評估，2026-06-06 |
| [evaluation_results.json](evaluation_results.json) | 評估回答與逐項檢查結果 |

## 解讀限制

評估 JSON 記錄 20 / 23 項檢查通過，包含 10 個問題。這個比例的分母是規則數，不是回答數，也不是外部資料庫核實後的正確率。部分規則只要求出現關鍵字或足夠長度，未提水溫時，水溫檢查也可能通過。

例如「95 分以上」的回答同時出現 `96/100` 和 `94.2/100`，仍被標為通過；目前整數正規表示式無法完整處理小數評分。產區比較中即使出現不可靠的說明，也可能因包含產區名稱而通過。原始結果保留，以便追溯這些限制。

`05_evaluation.py` 評估的是程式內固定題目，未使用 `data/processed/coffee_test.json`，也未核實推薦內容是否對應真實資料。互動推論的 `post_process()` 提醒未套用到該評估流程。A/B 生成使用抽樣且未固定生成 seed，需視為行為案例。

## 設定差異

歷史 A/B 報告及產生器寫有 LoRA rank 16、alpha 32、3 epochs；目前 `02_train.py` 為 rank 8、alpha 16、2 epochs。本機最外層 `adapter_config.json` 同樣記錄 rank 8、alpha 16，但歷史報告未附權重雜湊，因此無法確認每份報告使用的確切 checkpoint。

修訂版簡報收錄於 [docs/presentations/](../docs/presentations/)。`scripts/generate_ppt.py` 為另一份歷史簡報產生器，不能用來重建該修訂版。
