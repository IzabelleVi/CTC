import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
import librosa
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Load pretrained Wav2Vec 2.0 model 
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(model_name)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
# Set to evaluation mode
model.eval()

print("Model loaded successfully")

# Load the Common Voice dataset (English subset)
dataset_large = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test")
print("passes this sht atleast")
dataset = dataset_large.select(range(10))

print("Dataset loaded successfully")

def preprocess_audio(example):
    """Resamples audio to 16kHz for Wav2Vec 2.0"""
    waveform, sr = torchaudio.load(example["path"])
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    example["audio"] = resampler(waveform).squeeze().numpy()
    return example

# Apply preprocessing
dataset = dataset.map(preprocess_audio)

print("Preprocessing completed successfully")

def segment_speech(audio):
    """Runs Wav2Vec 2.0 CTC and detects word boundaries."""
    
    # Tokenize input
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    # Get logits from the model
    with torch.no_grad():
        logits = model(inputs.input_values).logits  # (batch, time_steps, vocab_size)

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get predicted token indices
    pred_ids = torch.argmax(probs, dim=-1)

    # Convert token IDs to text
    transcription = processor.batch_decode(pred_ids)[0]
    
    print("Predicted Transcription:")
    return transcription, logits, pred_ids

# Example: Run on a single test sample
audio_input = dataset[0]["audio"]
transcription, logits, pred_ids = segment_speech(audio_input)

print("succesfully Predicted Transcription:", transcription)

def get_word_boundaries(pred_ids, audio_len, sampling_rate=16000):
    """Finds word boundaries based on blank tokens in the CTC output."""

    # Get blank token ID (default is 0)
    blank_token_id = processor.tokenizer.pad_token_id

    # Find time steps where blank is predicted
    blank_frames = (pred_ids[0] == blank_token_id).cpu().numpy()

    # Compute time per step (in seconds)
    time_per_step = audio_len / len(blank_frames) / sampling_rate  

    # Get indices where blank tokens appear (word boundaries)
    boundaries = np.where(blank_frames[:-1] != blank_frames[1:])[0] * time_per_step

    return boundaries

# Compute word boundaries
word_boundaries = get_word_boundaries(pred_ids, len(audio_input))

print("Estimated Word Boundaries (seconds):", word_boundaries)

# Plot waveform
plt.figure(figsize=(12, 4))
plt.plot(audio_input, label="Waveform", alpha=0.7)


i = 0
# Plot word boundaries
for  boundary in word_boundaries:
    plt.axvline(x=boundary * 16000, color="red", linestyle="--", alpha=0.7, label="Word Boundary" if i == 0 else "")
    i+=1

plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.title("Word Segmentation using Wav2Vec2 CTC")
plt.legend()
plt.show()
