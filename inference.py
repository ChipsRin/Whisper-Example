import os
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from tqdm import tqdm

# ====== 參數設定 ======
model_path = "/path/to/model"
audio_folder = "/path/to/test_wav"
output_file = "transcriptions.txt" 
do_segment = False 

# ====== 載入 Processor 和 Model ======
try:
    processor = AutoProcessor.from_pretrained(model_path)
    print("成功載入 Processor 和 Tokenizer")
except:
    print("找不到 tokenizer，從原始模型下載並補回...")
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3turbo")
    processor.save_pretrained(model_path)
    print("已補回 Tokenizer！")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path).to(device)
model.eval()

# ====== 工具函數 ======
def segment_text(text):
    """將中文逐字分隔（保留空格）"""
    return " ".join(text)

def compute_entropy(probs):
    # probs: Tensor shape (seq_len, vocab_size)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # shape: (seq_len,)
    return entropy.mean().item()

def transcribe(audio_path):
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
        waveform = waveform.squeeze().numpy()

        # 特徵提取
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(device)

        # 產生 attention_mask（全為 1，表示無 padding）
        attention_mask = torch.ones(input_features.shape[:-1], dtype=torch.long).to(device)

        forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")

        with torch.no_grad():
           predicted_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                forced_decoder_ids=forced_decoder_ids,
                num_beams=5,              
                max_new_tokens=256,       
                length_penalty=1.0,        
                early_stopping=False,     
                no_repeat_ngram_size=3  
            )

        transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return segment_text(transcription) if do_segment else transcription

    except Exception as e:
        print(f"轉錄失敗：{audio_path}，錯誤：{e}")
        return None

def transcribe_folder(root_folder, output_path):
    """遞迴處理資料夾內所有音檔，並寫入輸出結果"""
    all_audio_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".wav"):
                all_audio_files.append(os.path.join(root, file))

    if not all_audio_files:
        print("找不到 .wav 檔案！")
        return

    print(f"開始轉錄 {len(all_audio_files)} 個音訊檔案...\n")

    with open(output_path, "w", encoding="utf-8") as f:
        for audio_path in tqdm(all_audio_files, desc="Processing"):
            transcription = transcribe(audio_path)
            if transcription:
                relative_path = os.path.relpath(audio_path, root_folder)
                file_id = os.path.splitext(relative_path)[0].replace(os.sep, "_")
                f.write(f"{file_id} {transcription}\n")

    print(f"所有音訊轉錄完成！結果已儲存至 `{output_path}`")

# ====== 主程式 ======
if __name__ == "__main__":
    transcribe_folder(audio_folder, output_file)
