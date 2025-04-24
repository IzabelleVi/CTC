import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
import ctc_segmentation
from sklearn.metrics import mean_absolute_error

# Load pretrained Dutch Wav2Vec2 model
processor = Wav2Vec2Processor.from_pretrained("./50-50 model")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./50-50 model")
model = Wav2Vec2ForCTC.from_pretrained("./50-50 model")
model.eval()

# Load audio file
audio_path = "timit_sample.wav"
wav, sr = torchaudio.load(audio_path)
resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
audio = resampler(wav).squeeze().numpy()

# Load word timestamps from .wrd file
def load_actual_timestamps(wrd_path):
    actual_timestamps = []
    with open(wrd_path, "r") as f:
        for line in f:
            start, end, word = line.strip().split()
            actual_timestamps.append({
                "text": word,
                "start": int(start) / 16000,  # Convert to seconds
                "end": int(end) / 16000
            })
    return actual_timestamps

actual_timestamps = load_actual_timestamps("timit_sample.wrd")

# Predict word timestamps using Wav2Vec2 model
def get_word_timestamps(audio):
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000, padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits.cpu()[0]
    predicted_ids = torch.argmax(logits, dim=-1)
    pred_transcript = processor.decode(predicted_ids)
    
    words = pred_transcript.split(" ")
    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = len(audio) / logits.shape[0] / 16000
    
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, words)
    timings, char_probs, _ = ctc_segmentation.ctc_segmentation(config, torch.nn.functional.softmax(logits, dim=-1).numpy(), ground_truth_mat)
    segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, words)
    
    return [{"text": w, "start": p[0], "end": p[1]} for w, p in zip(words, segments)]

predicted_timestamps = get_word_timestamps(audio)

# Compute mean absolute error between actual and predicted timestamps, handling different lengths
min_length = min(len(actual_timestamps), len(predicted_timestamps))
actual_start_times = [t["start"] for t in actual_timestamps[:min_length]]
predicted_start_times = [t["start"] for t in predicted_timestamps[:min_length]]

if min_length > 0:
    mae = mean_absolute_error(actual_start_times, predicted_start_times)
    print(f"Mean Absolute Error: {mae:.3f} seconds")
else:
    print("No matching words to compute error.")


# Plot waveform with actual and predicted word boundaries
plt.figure(figsize=(12, 6))
plt.plot(audio, label="Waveform", alpha=0.7)

for i, word in enumerate(actual_timestamps):
    plt.axvline(x=word["start"] * 16000, color="blue", linestyle="--", alpha=0.7, label="Actual" if i == 0 else "")
    plt.text(word["start"] * 16000, -1.2, word["text"], rotation=45, color="blue", fontsize=10, ha='right')

for i, word in enumerate(predicted_timestamps):
    plt.axvline(x=word["start"] * 16000, color="red", linestyle=":", alpha=0.7, label="Predicted" if i == 0 else "")
    plt.text(word["start"] * 16000, 1.2, word["text"], rotation=45, color="red", fontsize=10, ha='left')

plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.title("Word Segmentation: Actual vs Predicted")
plt.legend()
plt.show()

# Print model's predicted words
print("Predicted Transcript:", " ".join([t["text"] for t in predicted_timestamps]))
