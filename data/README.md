# 資料準備

資料來源：[Kaggle Coffee Reviews Dataset](https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset)。原始 CSV 與衍生對話不納入 Git，請自行下載並確認使用條款。

## 輸入

將 CSV 放在 `data/raw/`。程式依序優先尋找 `simplified_coffee.csv`、`coffee_analysis.csv`、`top-rated-coffee-clean.csv`，再搜尋子目錄下的其他 CSV，讀取第一個存在的候選檔。若有多個候選，請確認終端顯示的實際來源。

必要欄位為 `name`、`roast`、`rating`。為產生完整回答，建議同時包含 `roaster`、`origin_1`、`desc_1`、`desc_2`、`desc_3`。中文對話來自字典、關鍵字與模板轉換，並非人工逐筆翻譯或標註。

```powershell
python scripts/01_data_preparation.py
```

## 輸出與切分

| 路徑 | 用途 |
| --- | --- |
| `data/processed/coffee_train.json` | 訓練對話 |
| `data/processed/coffee_val.json` | 驗證 loss 與 checkpoint 選擇 |
| `data/processed/coffee_test.json` | 保留的測試對話；目前規則式評估腳本未讀取此檔 |

程式設定 `random.seed(42)`，按 CSV 的 `name` 唯一值切分為約 80% / 10% / 10%，再各自產生 QA。同名品項分到同一集合；不同名稱的相似品項不保證隔離。通用知識、數值門檻與額外推理模板只加到訓練集合。

相同 seed 不等於跨環境完全可重現：請保存 CSV 雜湊、實際輸入路徑、程式版本與產生的 JSON 雜湊。重新執行會覆寫三個輸出檔。
