import torch
import random
import numpy as np
import torchaudio
import librosa
from datasets import load_dataset, DatasetDict
from transformers import (Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments)
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from datasets import concatenate_datasets
from transformers import DataCollatorWithPadding
from huggingface_hub import notebook_login
import re

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# Load Common Voice Dutch and English datasets
dutch_dataset = load_dataset("mozilla-foundation/common_voice_13_0", "nl", split="train", trust_remote_code=True)
en_dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train", trust_remote_code=True)
dutch_dataset = dutch_dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
en_dataset = en_dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch
dutch_dataset = dutch_dataset.map(remove_special_characters)
en_dataset = en_dataset.map(remove_special_characters)
print("Datasets loaded successfully & unnecessary columns & accents removed")

# Limit dataset size to match
def limit_dataset(dataset, max_hours=10, sample_rate=16000):
    total_duration = 0.0
    selected_data = []
    for example in dataset:
        audio_array = example["audio"]["array"]
        duration = len(audio_array) / sample_rate / 3600  # Convert to hours
        if total_duration + duration > max_hours:
            break
        total_duration += duration
        selected_data.append(example)
    return DatasetDict({"train": dataset.select(range(len(selected_data)))})


en_dataset = limit_dataset(en_dataset, max_hours=10)
dutch_dataset = limit_dataset(dutch_dataset, max_hours=10)
print("Datasets limited successfully")

# Merge datasets
dataset = DatasetDict({"train": concatenate_datasets([en_dataset["train"], dutch_dataset["train"]])})
print("Datasets merged successfully")

# Load tokenizer and feature extractor
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=80, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
data_collator = DataCollatorWithPadding(processor)
print("Tokenizer and feature extractor loaded successfully")

# Preprocessing function
def preprocess_function(batch):
    audio = batch["audio"]
    # Convert to Double
    audio_array = np.array(audio["array"], dtype=np.float64)
    # Resample audio to 16000 Hz if necessary using librosa
    if audio["sampling_rate"] != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=audio["sampling_rate"], target_sr=16000)
 # Ensure the input values are properly padded or truncated
    input_values = feature_extractor(audio_array, sampling_rate=16000).input_values[0]
    batch["input_values"] = input_values
    batch["labels"] = tokenizer(batch["sentence"]).input_ids if "sentence" in batch else []
    return batch

dataset = dataset.map(preprocess_function, remove_columns=["audio", "sentence"])
print("Preprocessing completed successfully")

# Define model
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",  
    vocab_size=len(processor.tokenizer),
    gradient_checkpointing=True,
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2_dutch_english_5050_10_hours",
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=500,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    warmup_steps=500,
    max_steps=10000,  # Reduce max steps for smaller dataset
    logging_dir="./logs",
    logging_steps=100,
    fp16=True,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=processor,
    data_collator=data_collator,  # Add this line
)

print(dataset["train"].shape)

# Train model
trainer.train()
print("Model trained successfully")

# Save model
model.save_pretrained("./wav2vec2_dutch_english_5050_10_hours")
processor.save_pretrained("./wav2vec2_dutch_english_5050_10_hours")
print("Model saved successfully")