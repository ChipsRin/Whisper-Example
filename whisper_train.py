import os
import torch
import torchaudio
import numpy as np
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from copy import deepcopy
import argparse
from audiomentations import Compose, TimeStretch, Gain, PitchShift, AddGaussianSNR, AddBackgroundNoise, OneOf
import torchaudio.transforms as T
import random

# --------- 1. 參數解析 ---------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True, help="Path to training jsonl file")
    parser.add_argument("--eval_jsonl", type=str, required=True, help="Path to eval jsonl file")
    parser.add_argument("--output_dir", type=str, default="./whisper-finetune-output", help="Output dir for model and logs")
    parser.add_argument("--noise_dir", type=str, default=None, help="Path to background noise directory")
    parser.add_argument("--use_augmentation", action="store_true", help="Enable waveform + SpecAugment")
    return parser.parse_args()

# --------- 2. waveform augment ---------
def build_waveform_augment(noise_dir):
    return Compose([
        TimeStretch(min_rate=0.95, max_rate=1.05, p=0.5, leave_length_unchanged=False),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.1),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.1),
        OneOf([
            AddBackgroundNoise(
                sounds_path=noise_dir,
                min_snr_db=10.0,
                max_snr_db=20.0,
                noise_transform=None,
                p=1.0
            ),
            AddGaussianSNR(min_snr_db=5.0, max_snr_db=20.0, p=1.0),
        ], p=0.1),
    ])

spec_augment = torch.nn.Sequential(
    T.FrequencyMasking(freq_mask_param=10),
    T.TimeMasking(time_mask_param=20)
)

# --------- 3. 預處理 ---------
def prepare_example(example, processor, use_aug):
    speech_array, sr = torchaudio.load(example["audio_filepath"])
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        speech_array = resampler(speech_array)
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)
    speech_array = speech_array.squeeze().numpy()

    example["raw_audio"] = speech_array
    example["labels"] = processor.tokenizer(example["text"], max_length=448, truncation=True).input_ids

    if use_aug:
        example["aug_variant"] = random.randint(0, 999999)

    return example

# --------- 4. data collator ---------
def build_collator(model, processor, use_aug, waveform_aug):
    def collate_fn(batch):
        input_features_list = []
        labels_list = []
        training = model.training if hasattr(model, "training") else True

        for item in batch:
            raw_audio = np.array(item["raw_audio"])
            if use_aug and training:
                raw_audio = waveform_aug(samples=raw_audio, sample_rate=16000)

            input_features = processor.feature_extractor(raw_audio, sampling_rate=16000).input_features[0]
            input_tensor = torch.tensor(input_features)
            if use_aug and training:
                input_tensor = spec_augment(input_tensor.unsqueeze(0)).squeeze(0)

            input_features_list.append(input_tensor)
            labels_list.append(torch.tensor(item["labels"], dtype=torch.long))

        input_features_padded = torch.nn.utils.rnn.pad_sequence(input_features_list, batch_first=True, padding_value=0)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)

        return {"input_features": input_features_padded, "labels": labels_padded}

    return collate_fn

# --------- 5. loss curve callback ---------
class LossPlotCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.train_losses.append((logs.get("epoch", 0), logs["loss"]))
            if "eval_loss" in logs:
                self.eval_losses.append((logs.get("epoch", 0), logs["eval_loss"]))

    def on_train_end(self, args, state, control, **kwargs):
        import matplotlib.pyplot as plt
        if self.train_losses:
            epochs, values = zip(*self.train_losses)
            plt.plot(epochs, values, label="Train Loss")
        if self.eval_losses:
            epochs, values = zip(*self.eval_losses)
            plt.plot(epochs, values, label="Eval Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.title("Loss Curve (Epoch-based)")
        plt.savefig(os.path.join(args.output_dir, "loss_curve_epoch.png"))
        plt.close()

# --------- 6. main ---------
if __name__ == "__main__":
    args = parse_args()

    dataset = load_dataset("json", data_files={"train": args.train_jsonl, "eval": args.eval_jsonl})
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo", language="Chinese", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")

    waveform_aug = build_waveform_augment(args.noise_dir) if args.use_augmentation else None

    dataset["train"] = dataset["train"].map(lambda x: prepare_example(x, processor, args.use_augmentation), num_proc=8)
    dataset["eval"] = dataset["eval"].map(lambda x: prepare_example(x, processor, False), num_proc=8)

    data_collator = build_collator(model, processor, args.use_augmentation, waveform_aug)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=15,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=10,
        learning_rate=1e-5,
        warmup_steps=500,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
        evaluation_strategy="steps",
        eval_steps=200,
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        callbacks=[LossPlotCallback(), EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "best_model"))
    processor.save_pretrained(training_args.output_dir)
    processor.tokenizer.save_pretrained(training_args.output_dir)
