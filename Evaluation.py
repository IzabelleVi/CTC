import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
import ctc_segmentation
from sklearn.metrics import precision_recall_fscore_support

# Load pretrained Wav2Vec 2.0 model 
processor = Wav2Vec2Processor.from_pretrained("./Dutch only model")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./Dutch only model")
model = Wav2Vec2ForCTC.from_pretrained("./Dutch only model")

# Set model to evaluation mode
model.eval()
print("Model loaded successfully")

# Load the Common Voice dataset (English subset)
dataset_large = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test", trust_remote_code=True)
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
    """Runs Wav2Vec 2.0 CTC and gets word timestamps using CTC segmentation."""
    
    # Tokenize input
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    # Get logits from the model
    with torch.no_grad():
        logits = model(inputs.input_values).logits  # (batch, time_steps, vocab_size)

    # Convert logits to probabilities
    pred_ids = torch.argmax(logits, dim=-1)

    # Convert token IDs to text
    transcription = processor.batch_decode(pred_ids)[0]

    print("Predicted Transcription:", transcription)

    # Change audio format. Wav2Vec2 uses float32, but ctc_segmentation uses float64

    # Min-Max scaling to range [0, 1]
    min_val = np.min(audio)
    max_val = np.max(audio)
    scaled_data = (audio - min_val) / (max_val - min_val)

    # If you need to scale to a different range, e.g., [0, 1], you can adjust the formula accordingly
    # For example, to scale to [0, 1]:
    desired_range = (0, 1)
    scaled_data = desired_range[0] + (scaled_data * (desired_range[1] - desired_range[0]))

    # Get word timestamps using CTC segmentation
    word_timestamps = get_word_timestamps(scaled_data)

    return transcription, word_timestamps

def get_word_timestamps(audio, samplerate=16000):
    assert audio.ndim == 1
    
    # Run prediction, get logits and probabilities
    inputs = processor(audio, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = model(inputs.input_values).logits.cpu()[0]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
    predicted_ids = torch.argmax(logits, dim=-1)
    pred_transcript = processor.decode(predicted_ids)

    # Split the transcription into words
    words = pred_transcript.split(" ")

    # Align
    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio.shape[0] / probs.size()[0] / samplerate

    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, words)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
    segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, words)

    return [{"text": w, "start": p[0], "end": p[1], "conf": p[2]} for w, p in zip(words, segments)]

# Example: Run on a single test sample
audio_input = dataset[0]["audio"]
transcription, word_timestamps = segment_speech(audio_input)

print("Word Timestamps:", word_timestamps)

# Plot waveform with word timestamps
plt.figure(figsize=(12, 4))
plt.plot(audio_input, label="Waveform", alpha=0.7)

for i, word in enumerate(word_timestamps):
    plt.axvline(x=word["start"] * 16000, color="red", linestyle="--", alpha=0.7, label="Word Boundary" if i == 0 else "")

plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.title("Word Segmentation using Wav2Vec2 CTC")
plt.legend()
plt.show()
