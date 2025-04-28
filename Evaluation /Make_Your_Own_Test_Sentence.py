import sounddevice as sd
import soundfile as sf
import pygame
import time
import threading
import json

def record_audio(filename, duration, samplerate=44100):
    print("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=2)
    sd.wait()
    sf.write(filename, recording, samplerate)
    print(f"Recording saved as {filename}")

def play_audio(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

def collect_timestamps():
    timestamps = []
    print("Press SPACE to add a timestamp. Press ESC to finish.")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    timestamps.append(time.time() - start_time)
                    print(f"Timestamp added at {timestamps[-1]:.2f} seconds")
                elif event.key == pygame.K_ESCAPE:
                    running = False
    return timestamps

def main():
    audio_filename = "recording.wav"
    timestamps_filename = "timestamps.json"
    transcript_filename = "transcript.txt"

    duration = int(input("Enter recording duration (seconds): "))
    record_audio(audio_filename, duration)

    print("\nGet ready to timestamp...")
    input("Press Enter to start playback...")

    global start_time
    start_time = time.time()
    
    # Start playback
    play_audio(audio_filename)

    # Collect timestamps
    timestamps = collect_timestamps()

    # Save timestamps
    with open(timestamps_filename, 'w') as f:
        json.dump(timestamps, f, indent=4)
    print(f"Timestamps saved to {timestamps_filename}")

    # Get transcript
    print("\nNow write the transcript. End input with an empty line:")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    transcript = "\n".join(lines)

    # Save transcript
    with open(transcript_filename, 'w') as f:
        f.write(transcript)
    print(f"Transcript saved to {transcript_filename}")

if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()
