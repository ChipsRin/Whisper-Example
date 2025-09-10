# Whisper 模型微調訓練範例

本文件說明如何使用 `whisper_train.py` 進行 Whisper 模型的微調訓練。

## 環境設定

### 建立 Conda 環境

```bash
# 建立新環境
conda create -n whisper_train python=3.10 -y
conda activate whisper_train

# 安裝 PyTorch (CUDA 版本，根據系統 CUDA 版本調整)
conda install pytorch torchaudio pytorch-cuda -c pytorch -c nvidia

# 安裝其他必需套件
pip install transformers datasets accelerate evaluate jiwer
pip install audiomentations matplotlib tqdm
```

## 腳本概覽

基於 Whisper 系列模型的微調訓練腳本，支援完整的資料增強和訓練監控功能。

### 核心特性

- **基礎模型**: 支援所有 Whisper 模型大小 (tiny, base, small, medium, large-v2, large-v3, large-v3-turbo)
- **資料增強**: 波形級 + 頻譜級雙重增強
- **訓練監控**: 自動保存最佳模型、Loss 曲線繪製
- **早停機制**: 防止過擬合，提升泛化能力

## 基本使用方法

### 必需參數

```bash
python whisper_train.py \
    --train_jsonl train.jsonl \        # 訓練資料檔案路徑
    --eval_jsonl eval.jsonl \          # 驗證資料檔案路徑
    --output_dir ./my_model            # 輸出目錄
```

### 完整參數

```bash
python whisper_train.py \
    --train_jsonl train.jsonl \
    --eval_jsonl eval.jsonl \
    --output_dir ./my_model \
    --noise_dir /path/to/noise \       # 背景噪音檔案目錄(可選)
    --use_augmentation                 # 啟用資料增強(建議)
```

## 資料準備

### JSONL 檔案格式

每行包含音檔路徑和對應文字：

```json
{"audio_filepath": "path/to/audio1.wav", "text": "這是第一段語音的內容"}
{"audio_filepath": "path/to/audio2.wav", "text": "這是第二段語音的內容"}
{"audio_filepath": "path/to/audio3.wav", "text": "這是第三段語音的內容"}
```

### 音檔要求

- **格式**: WAV, MP3, FLAC (建議 WAV)
- **採樣率**: 任意 (自動重新採樣至 16kHz)
- **聲道**: 單聲道或雙聲道 (自動轉換為單聲道)
- **時長**: 建議 1-30 秒

## 訓練配置

### 批次大小設定

```python
per_device_train_batch_size=15      # 每 GPU 訓練批次大小
per_device_eval_batch_size=2        # 每 GPU 驗證批次大小  
gradient_accumulation_steps=2       # 梯度累積步數
```

**有效批次大小** = `per_device_train_batch_size × gradient_accumulation_steps × GPU數量`

### 學習率和訓練輪數

```python
learning_rate=1e-5                  # 學習率
warmup_steps=500                    # 預熱步數
num_train_epochs=10                 # 最大訓練輪數
```

### 模型保存策略

```python
save_steps=200                      # 每 200 步保存一次
save_total_limit=2                  # 最多保存 2 個檢查點
load_best_model_at_end=True         # 訓練結束載入最佳模型
metric_for_best_model="eval_loss"   # 最佳模型評判標準
```

## 資料增強功能

### 啟用方式

```bash
--use_augmentation                  # 啟用資料增強
--noise_dir ./background_noise      # 背景噪音目錄(可選)
```

### 波形級增強

1. **時間拉伸**: 95%-105% 速度變化 (50% 機率)
2. **音量調整**: ±6dB 增益變化 (10% 機率)  
3. **音調變化**: ±2 半音變化 (10% 機率)
4. **噪音添加**: SNR 10-20dB (10% 機率)

### 頻譜級增強 (SpecAugment)

- **頻率遮蔽**: 最多遮蔽 10 個頻率 bins
- **時間遮蔽**: 最多遮蔽 20 個時間 frames

## 訓練輸出

### 目錄結構

```
output_dir/
├── best_model/                     # 最佳模型檔案
│   ├── config.json
│   ├── model.safetensors
│   ├── generation_config.json
│   └── tokenizer files
├── logs/                           # TensorBoard 日誌
├── loss_curve_epoch.png           # Loss 曲線圖
└── checkpoint-xxx/                 # 訓練檢查點
```

### 關鍵檔案

- **best_model/**: 驗證 Loss 最低的模型
- **loss_curve_epoch.png**: 訓練和驗證 Loss 曲線
- **logs/**: TensorBoard 監控檔案

## 使用範例

### 基本微調

```bash
# 準備資料
echo '{"audio_filepath": "./audio/sample1.wav", "text": "Hello world"}' > train.jsonl
echo '{"audio_filepath": "./audio/sample2.wav", "text": "Good morning"}' >> train.jsonl

echo '{"audio_filepath": "./audio/eval1.wav", "text": "Test sample"}' > eval.jsonl

# 開始訓練
python whisper_train.py \
    --train_jsonl train.jsonl \
    --eval_jsonl eval.jsonl \
    --output_dir ./whisper_finetuned
```

### 完整微調 (含資料增強)

```bash
# 訓練含資料增強
python whisper_train.py \
    --train_jsonl train.jsonl \
    --eval_jsonl eval.jsonl \
    --output_dir ./whisper_finetuned_aug \
    --use_augmentation \
    --noise_dir ./noise_samples
```

### 更換不同大小的 Whisper 模型

修改 `whisper_train.py` 第 132-133 行來選擇不同大小的模型：

```python
# 範例：使用 small 模型
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Chinese", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
```

## 監控訓練

### TensorBoard

```bash
tensorboard --logdir ./whisper_finetuned/logs
```

在瀏覽器開啟 `http://localhost:6006` 查看訓練曲線。

### 重要指標

- **train_loss**: 訓練損失，應持續下降
- **eval_loss**: 驗證損失，用於早停和最佳模型選擇
- **learning_rate**: 學習率變化曲線

## 參數調整建議

### 記憶體不足

```python
per_device_train_batch_size=8       # 減少批次大小
per_device_eval_batch_size=1        
gradient_accumulation_steps=4       # 增加梯度累積
```

### 防止過擬合

```python
# 啟用資料增強
--use_augmentation

# 調整早停參數
EarlyStoppingCallback(early_stopping_patience=3)
```

## 繼續訓練

如需從檢查點繼續訓練，修改 `TrainingArguments`:

```python
training_args = TrainingArguments(
    # ... 其他參數
    resume_from_checkpoint="./output_dir/checkpoint-xxx"
)
```

## 模型推理

訓練完成後使用 `inference.py` 進行推理：

```python
# 修改推理腳本參數
model_path = "./whisper_finetuned/best_model"
audio_folder = "/path/to/test_audio"
output_file = "transcriptions.txt"
```

---

**提示**: 建議先用小量資料測試腳本功能，確認無誤後再進行大規模訓練。