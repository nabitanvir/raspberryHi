import config, utils
import os, time
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

def record_audio(sample_rate, duration, device=None):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16', device=device)
    sd.wait()
    return recording.flatten()

def main():
    save_directory = config.POSITIVE_DIRECTORY
    chosen_dir = input("Positive or negative recordings? (p/n): ")
    if chosen_dir == 'p':
        save_directory = config.POSITIVE_DIRECTORY
    elif chosen_dir == 'n':
        save_directory = config.NEGATIVE_DIRECTORY
    else:
        print("Directory not chosen, terminating...")
        exit()

    num_recordings = int(input("# of recordings: "))
    
    print("Press enter to start each recording\n")
    for i in range(num_recordings):
        input(f"Press Enter to start recording {i + 1}/{num_recordings}")
        audio_data = record_audio(config.SAMPLE_RATE, config.DURATION, device=config.AUDIO_DEVICE_ID)
        timestamp = int(time.time() * 1000)
        filename = f"berry_{timestamp}.wav"
        filepath = os.path.join(save_directory, filename)
        write(filepath, config.SAMPLE_RATE, audio_data)
        print(f"saved recording {i + 1} as {filepath}\n")

    print("Recording session complete, terminating...")

if __name__ == "__main__":
    main()
