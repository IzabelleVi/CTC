import torch
import random
import numpy as np
import torchaudio
from datasets import load_dataset
from transformers import (Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments, DataCollatorWithPadding)

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# Load Common Voice dataset (single language: Dutch)
dataset = load_dataset("mozilla-foundation/common_voice_13_0", "nl", split="train", trust_remote_code=True)
dataset = dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

#en_dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train", trust_remote_code=True)
#en_dataset = en_dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

# Load the pre-trained Wav2Vec2 processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# Function to preprocess audio samples
def preprocess_function(batch):
    audio = batch["audio"]
    waveform = torch.tensor(audio["array"]).float()
    sampling_rate = audio["sampling_rate"]
    
    # Ensure the sampling rate is 16000
    if sampling_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(waveform)
        sampling_rate = 16000
    
    batch["input_values"] = processor(waveform, sampling_rate=sampling_rate).input_values[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

# Preprocess datasets
dataset = dataset.map(preprocess_function, remove_columns=["audio", "sentence"])
#en_dataset = en_dataset.map(preprocess_function, remove_columns=["audio", "sentence"])

# Define data collator
data_collator = DataCollatorWithPadding(processor)

# Load Wav2Vec2 model
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base", 
                                       attention_dropout=0.1,
                                       hidden_dropout=0.1,
                                       feat_proj_dropout=0.1,
                                       mask_time_prob=0.05,
                                       layerdrop=0.1,
                                       ctc_loss_reduction="mean",
                                       pad_token_id=processor.tokenizer.pad_token_id)

# Training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-dutch",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()
print("Training completed successfully")
# Save the fine-tuned model
trainer.save_model("./wav2vec2-dutch-final")