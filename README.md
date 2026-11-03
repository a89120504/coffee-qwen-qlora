# ☕ Coffee Qwen QLoRA — 咖啡品鑑助手

以 **Qwen2.5-3B-Instruct + QLoRA** 建立繁體中文咖啡品鑑助手，提供風味介紹、偏好推薦、產區說明與沖煮建議。資料準備程式將 Coffee Reviews CSV 轉為中文對話，並提供互動推論、微調前後比較與規則式評估。

本儲存庫收錄程式、操作文件、既有評估報告及修訂版簡報。資料集、處理後對話、模型權重與本機環境另行保存，clone 後須自行準備資料與 LoRA adapter。

## 專案結構

```text
coffee-qwen-qlora/
├── data/README.md                  # 資料來源、欄位與切分方式
├── docs/
│   ├── reproducibility.md          # 設定、環境盤點與重現限制
│   └── presentations/              # 修訂版期末簡報
├── models/README.md                # Adapter 路徑與準備方式
├── results/                        # 歷史 A/B 與規則式評估報告
├── scripts/
│   ├── 01_data_preparation.py      # CSV → 中文對話與資料切分
│   ├── 02_train.py                 # QLoRA 微調
│   ├── 03_inference.py             # 互動問答與規則提醒
│   ├── 04_ab_comparison.py         # 微調前後比較
│   ├── 04_ab_comparison_full.py    # 完整逐字比較
│   ├── 05_evaluation.py            # 固定題目的規則式檢查
│   └── generate_ppt.py             # 歷史簡報產生器
├── test_import.py                  # 本機依賴匯入診斷
├── requirements.txt               # 預設 Hugging Face 訓練路徑
├── requirements-unsloth.txt        # 選用的 Unsloth 後端
└── requirements-presentation.txt   # 選用的簡報產生套件
```

## 環境設定

以下以 Windows PowerShell 為例，從專案根目錄執行。訓練及 4-bit 推論需要可用的 CUDA GPU；本次整理未驗證完整 GPU 流程。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 依 GPU 與驅動選擇官方 PyTorch CUDA 安裝指令，再安裝專案依賴
python -m pip install -r requirements.txt
python -m pip check
python test_import.py
```

PyTorch 安裝方式請參考[官方安裝頁面](https://pytorch.org/get-started/locally/)。`requirements.txt` 是依程式 API 整理的版本範圍，尚未在全新環境完成訓練驗證，並非歷史實驗的鎖定環境；詳見[環境與重現紀錄](docs/reproducibility.md)。

`02_train.py` 預設 `USE_UNSLOTH = False`，使用 Hugging Face 路徑。若要嘗試選用後端，先確認 Unsloth 的平台與版本相容性，再安裝 `requirements-unsloth.txt` 並手動切換該常數；本次未驗證這條路徑。

## 執行流程

先依 [data/README.md](data/README.md) 下載 CSV，放入 `data/raw/`。

```powershell
# 1. 產生 train / validation / test 對話資料
python scripts/01_data_preparation.py

# 2. 訓練 LoRA adapter
python scripts/02_train.py

# 3. 互動問答（輸入 quit 結束）
python scripts/03_inference.py

# 4. 生成微調前後比較
python scripts/04_ab_comparison.py
python scripts/04_ab_comparison_full.py

# 5. 執行固定題目的規則式評估
python scripts/05_evaluation.py
```

這些腳本目前使用固定輸出路徑。重新執行資料準備、訓練或報告產生前，請先保存既有資料、adapter 與報告。推論所需檔案見 [models/README.md](models/README.md)。

## 目前程式設定

以下取自 `scripts/02_train.py`，不代表所有歷史報告採用相同設定。

| 項目 | 設定 |
| --- | --- |
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| 預設後端 | Hugging Face Transformers / PEFT / TRL |
| 量化 | 4-bit NF4、double quantization、FP16 計算 dtype |
| LoRA | rank 8、alpha 16、dropout 0.05 |
| Target modules | `q_proj`、`v_proj` |
| Learning rate | `2e-4` |
| Batch size | 每裝置 2，gradient accumulation 4 |
| Epochs / sequence length | 2 / 512 |
| Training precision | CUDA 支援 BF16 時使用 BF16，否則 FP16 |
| Seed | 42 |
| Model selection | 有 validation 資料時，依 `eval_loss` 載入最佳 checkpoint |

## 既有成果與解讀

- [A/B 精選報告](results/ab_comparison.md)
- [A/B 完整逐字報告](results/ab_comparison_full.md)
- [規則式評估報告](results/evaluation_report.md)與[原始結果 JSON](results/evaluation_results.json)
- [修訂版期末簡報](docs/presentations/coffee-qwen-final-presentation-revised.pptx)

既有 JSON 記錄 10 個問題、23 項規則檢查，其中 20 項通過（約 87.0%）。這是**規則檢查通過率**，不能當作咖啡知識或推薦事實的正確率。檢查包含關鍵字、格式與部分數值比對，未對照 CSV 核實品項、產地或評分；回答即使互相矛盾，也可能通過。

`05_evaluation.py` 使用程式內的固定問題，並未讀取 `coffee_test.json`。A/B 報告採取抽樣生成，不能據此宣稱穩定的量化改善。部分歷史報告與產生器仍寫有 rank 16、alpha 32、3 epochs，與目前程式及本機 adapter 設定不同；原始成果保留，差異列於 [results/README.md](results/README.md)。

## 簡報工具

```powershell
python -m pip install -r requirements-presentation.txt
python scripts/generate_ppt.py
```

此腳本會寫入 `results/Coffee_Sommelier_Presentation.pptx`，內容含歷史硬編碼設定，與 `docs/presentations/` 中的修訂版簡報不同，使用前需核對內容。

## 授權與來源

Base model：[Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)。資料來源：[Kaggle Coffee Reviews Dataset](https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset)。本儲存庫不重新散布資料集或模型權重；使用時請分別確認來源的授權與使用條款。專案尚未指定程式碼開源授權。
