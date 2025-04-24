import torch
import random
import numpy as np
import librosa
from datasets import load_dataset, DatasetDict
from transformers import (Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments)
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from datasets import concatenate_datasets
from transformers import DataCollatorWithPadding
import re

model = Wav2Vec2ForCTC.from_pretrained("./wav2vec-dutch")

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
#en_dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train", trust_remote_code=True)
dutch_dataset = dutch_dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
#en_dataset = en_dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch
dutch_dataset = dutch_dataset.map(remove_special_characters)
#en_dataset = en_dataset.map(remove_special_characters)
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


#en_dataset = limit_dataset(en_dataset, max_hours=10)
dutch_dataset = limit_dataset(dutch_dataset, max_hours=10)
print("Datasets limited successfully")

# Merge datasets
#dataset = DatasetDict({"train": concatenate_datasets([dutch_dataset["train"], dutch_dataset["train"]])})
#print("Datasets merged successfully")

# Load tokenizer and feature extractor
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=80, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
data_collator = DataCollatorWithPadding(tokenizer=processor.tokenizer, padding=True)
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
    # Use processor instead of feature extractor to generate input_values & attention_mask
    max_length = 16000 * 11  # change the 11 to however many seconds you want to limit the audio to
    inputs = processor(audio_array, sampling_rate=16000, padding=True, truncation=True, max_length=max_length)  # Add max_length here
    # Assign input_values directly, no need for tolist()
    batch["input_values"] = inputs.input_values[0]  # Extract the list of audio samples from the list

    with processor.as_target_processor():  
        batch["labels"] = tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=128).input_ids
          # Generate input_ids using the processor for the input text
    batch["input_ids"] = tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=128).input_ids # Add padding="max_length", truncation and max_length for input_ids

    return batch

# Apply padding and truncation during preprocessing
dataset = dutch_dataset.map(preprocess_function, remove_columns=["audio"]) # We will use padding here instead.
print("Preprocessing completed successfully")

# Define model
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",  
    vocab_size=len(processor.tokenizer),
    gradient_checkpointing=True,
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-dutch-iza-10hours",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,  # Adjust if needed
    learning_rate=3e-4,
    warmup_steps=500,
    max_steps=10000,  # Adjust based on your needs
    fp16=torch.cuda.is_available(),
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    do_train=True,
    do_eval=False,  # No evaluation step
    report_to="none",  # Disable logging to WandB or other platforms
    dataloader_num_workers=4,
    remove_unused_columns=False,  # Important for audio input
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
)

print(dataset["train"].shape)

# Train model
trainer.train()
print("Model trained successfully")

# Save model
model.save_pretrained("./wav2vec2_dutch_english_5050_10_hours")
processor.save_pretrained("./wav2vec2_dutch_english_5050_10_hours")
print("Model saved successfully")