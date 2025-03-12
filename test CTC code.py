import torch
import numpy as np
from typing import List
import ctc_segmentation
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

# load model, processor and tokenizer
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(model_name)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# load dummy dataset and read soundfiles
SAMPLERATE = 16000
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
audio = ds[0]["audio"]["array"]
print(ds[0]["audio"]["array"])
transcripts = ["A MAN SAID TO THE UNIVERSE", "SIR I EXIST"]

def align_with_transcript(
    audio : np.ndarray,
    transcripts : List[str],
    samplerate : int = SAMPLERATE,
    model : Wav2Vec2ForCTC = model,
    processor : Wav2Vec2Processor = processor,
    tokenizer : Wav2Vec2CTCTokenizer = tokenizer
):
    assert audio.ndim == 1
    # Run prediction, get logits and probabilities
    inputs = processor(audio, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = model(inputs.input_values).logits.cpu()[0]
        probs = torch.nn.functional.softmax(logits,dim=-1)
    
    # Tokenize transcripts
    vocab = tokenizer.get_vocab()
    inv_vocab = {v:k for k,v in vocab.items()}
    unk_id = vocab["<unk>"]
    
    tokens = []
    for transcript in transcripts:
        assert len(transcript) > 0
        tok_ids = tokenizer(transcript.replace("\n"," ").lower())['input_ids']
        tok_ids = np.array(tok_ids,dtype=np.int64)
        tokens.append(tok_ids[tok_ids != unk_id])
    
    # Align
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio.shape[0] / probs.size()[0] / samplerate
    
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(config, tokens)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
    segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, transcripts)
    return [{"text" : t, "start" : p[0], "end" : p[1], "conf" : p[2]} for t,p in zip(transcripts, segments)]
    
def get_word_timestamps(
    audio : np.ndarray,
    samplerate : int = SAMPLERATE,
    model : Wav2Vec2ForCTC = model,
    processor : Wav2Vec2Processor = processor,
    tokenizer : Wav2Vec2CTCTokenizer = tokenizer
):
    assert audio.ndim == 1
    # Run prediction, get logits and probabilities
    inputs = processor(audio, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = model(inputs.input_values).logits.cpu()[0]
        probs = torch.nn.functional.softmax(logits,dim=-1)
        
    predicted_ids = torch.argmax(logits, dim=-1)
    pred_transcript = processor.decode(predicted_ids)
    
    # Split the transcription into words
    words = pred_transcript.split(" ")
    
    # Align
    vocab = tokenizer.get_vocab()
    inv_vocab = {v:k for k,v in vocab.items()}
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio.shape[0] / probs.size()[0] / samplerate
    
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, words)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
    segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, words)
    return [{"text" : w, "start" : p[0], "end" : p[1], "conf" : p[2]} for w,p in zip(words, segments)]

print(align_with_transcript(audio,transcripts))
# [{'text': 'A MAN SAID TO THE UNIVERSE', 'start': 0.08124999999999993, 'end': 2.034375, 'conf': 0.0}, 
#  {'text': 'SIR I EXIST', 'start': 2.3260775862068965, 'end': 4.078771551724138, 'conf': 0.0}]

print(get_word_timestamps(audio))
# [{'text': 'a', 'start': 0.08124999999999993, 'end': 0.5912715517241378, 'conf': 0.9999501323699951}, 
# {'text': 'man', 'start': 0.5912715517241378, 'end': 0.9219827586206896, 'conf': 0.9409108982174931}, 
# {'text': 'said', 'start': 0.9219827586206896, 'end': 1.2326508620689656, 'conf': 0.7700278702302796}, 
# {'text': 'to', 'start': 1.2326508620689656, 'end': 1.3529094827586206, 'conf': 0.5094435178226225}, 
# {'text': 'the', 'start': 1.3529094827586206, 'end': 1.4831896551724135, 'conf': 0.4580493446392211}, 
# {'text': 'universe', 'start': 1.4831896551724135, 'end': 2.034375, 'conf': 0.9285054256219009}, 
# {'text': 'sir', 'start': 2.3260775862068965, 'end': 3.036530172413793, 'conf': 0.0}, 
# {'text': 'i', 'start': 3.036530172413793, 'end': 3.347198275862069, 'conf': 0.7995760873559864}, 
# {'text': 'exist', 'start': 3.347198275862069, 'end': 4.078771551724138, 'conf': 0.0}]