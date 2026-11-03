# 模型檔案

模型權重、tokenizer 與 checkpoint 不納入 Git。推論與 A/B 腳本預期 adapter 位於：

```text
models/coffee-qwen2.5-3b-lora/
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer files saved by the training script
```

可使用 `scripts/02_train.py` 自行訓練，或將可信來源、相容的 adapter 放到上述路徑。目前沒有提供另外上傳的權重下載連結。

推論腳本會從 `Qwen/Qwen2.5-3B-Instruct` 載入 base model 及 tokenizer，再疊加本機 adapter。請確認網路／模型快取、CUDA 與量化套件可用。重新訓練會使用同一模型輸出目錄，請先保存既有權重。
