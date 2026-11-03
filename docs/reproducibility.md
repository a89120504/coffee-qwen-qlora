# 環境與重現紀錄

盤點日期：2026-11-03。以下區分目前程式、現存檔案與本次實際執行結果，不回推歷史實驗的完整條件。

## 目前程式

- `01_data_preparation.py`：按咖啡名稱分組切分，seed 42；只在 train 加入通用知識與額外模板。
- `02_train.py`：Hugging Face 後端、rank 8、alpha 16、2 epochs、sequence length 512、seed 42；完整設定見根目錄 README 與原始碼。
- `04_ab_comparison*.py`：抽樣生成，未固定生成 seed；報告標頭仍包含歷史硬編碼設定。
- `05_evaluation.py`：固定問題與字串規則，並保留既有 Windows import workaround。未讀取保留的 test JSON。

## 本機套件盤點

透過既有 `venv/Scripts/python.exe` 的 `importlib.metadata` 讀取套件版本；這不是成功載入所有套件的證明，也不是歷史訓練環境的 lockfile。

| 套件 | 盤點版本 |
| --- | --- |
| Python | 3.12.3 |
| torch | 2.5.1+cu124 |
| torchvision | 0.20.1+cu124 |
| transformers | 4.46.3 |
| datasets | 4.8.5 |
| trl | 0.12.2 |
| peft | 0.13.2 |
| bitsandbytes | 0.49.2 |
| accelerate | 1.13.0 |
| pandas | 3.0.3 |
| scikit-learn | 1.8.0 |
| python-pptx | 1.0.2 |
| unsloth | 2025.3.19 |

目前訓練程式使用 `SFTConfig(max_length=..., eos_token=...)`，本機 TRL 0.12.2 的設定類別仍使用 `max_seq_length`，不支援這些參數。依 [TRL 0.24.0 官方 API](https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer)與 [PyPI 套件宣告](https://pypi.org/pypi/trl/0.24.0/json)，將 `requirements.txt` 的 TRL 範圍收斂為 `>=0.24.0,<0.25.0`，並對齊其 Transformers、Datasets 與 Accelerate 下限。

這是配合現有程式的依賴宣告修正，未安裝或升級本機環境，也未完成全新環境的端到端驗證。Unsloth 從預設依賴分出，避免預設 Hugging Face 路徑要求安裝選用後端。

## 現存產物

本次只讀取檔案，沒有重新切分、訓練或評估。

| 產物 | 觀察 |
| --- | --- |
| `coffee_train.json` | 10,248 筆對話 |
| `coffee_val.json` | 1,260 筆對話 |
| `coffee_test.json` | 1,230 筆對話 |
| 外層 adapter config | rank 8、alpha 16、`q_proj` / `v_proj` |
| `checkpoint-1872/trainer_state.json` | epoch 2.0、global step 1872、最佳步數 1800、最佳紀錄值約 0.253613 |

對話數、checkpoint 狀態與歷史報告之間缺少完整雜湊及命令紀錄，不能據此建立確切的一對一來源關係。資料與權重保留於本機，不納入儲存庫。

## 本次驗證界線

在既有虛擬環境執行 `python -B test_import.py`，程序以 exit code 1 結束，未取得可用的匯入診斷輸出；不能宣稱目前環境可正常訓練。未下載 base model、未安裝套件、未執行 GPU 訓練或生成。

本次另外檢查程式語法、文件本機連結、Git 納入清單與既有評估 JSON 的加總一致性。這些檢查不能替代訓練或推論驗證。
