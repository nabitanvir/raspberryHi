import sounddevice as sd
import numpy as np
import os
import time
from scipy.io.wavfile import write

SAMPLE_RATE = 16000
DURATION = 1.0
POSITIVE_OUTPUT_DIR = 'datasets/wake_word/positive'
NEGATIVE_OUTPUT_DIR = 'datasets/wake_word/negative'

def record_audio(sample_rate, duration, device=None):
    print(f"Recording for {duration} seconds")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16', device=device)
    sd.wait()
    return recording.flatten()

def main():
    save_directory = 'datasets/wake_word/positive'
    chosen_dir = input("Positive or negative recordings? (p/n)")
    if chosen_dir == 'p':
        save_directory = 'datasets/wake_word/positive'
    elif chosen_dir == 'n':
        save_directory = 'datasets/wake_word/negative'
    else:
        print("Directory not chosen, ending session")
        exit()

    num_recordings = int(input("Enter the number of recordings you want to make: "))

    print("Get ready to record your wake word. Press Enter to start each recording.")
    for i in range(num_recordings):
        input(f"\nPress Enter to start recording {i + 1}/{num_recordings}")
        audio_data = record_audio(SAMPLE_RATE, DURATION, device=0)
        timestamp = int(time.time() * 1000)
        filename = f"berry_{timestamp}.wav"
        filepath = os.path.join(save_directory, filename)
        write(filepath, SAMPLE_RATE, audio_data)
        print(f"Saved recording {i + 1} as {filepath}")

    print("\nRecording session complete!")

if __name__ == "__main__":
    main()
