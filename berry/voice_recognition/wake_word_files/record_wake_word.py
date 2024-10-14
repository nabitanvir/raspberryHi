# This file will allow you to record your own voice in terminal and automatically save it in the positive/negative wake word directories
# I hardcoded my microphone ID in the code as "device=5", you can find your own microphone ID using "sd.query_devices()" in a python terminal
import os, time
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

SAMPLE_RATE = 16000
DURATION = 1.0

def record_audio(sample_rate, duration, device=None):
    print(f"recording for {duration} seconds")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16', device=device)
    sd.wait()
    return recording.flatten()

def main():
    save_directory = '/usr/src/app/berry/berry_voice_recognition/datasets/wake_word/positive'
    chosen_dir = input("positive or negative recordings? (p/n): ")
    if chosen_dir == 'p':
        save_directory = '/usr/src/app/berry/berry_voice_recognition/datasets/wake_word/positive'
    elif chosen_dir == 'n':
        save_directory = '/usr/src/app/berry/berry_voice_recognition/datasets/wake_word/negative'
    else:
        print("directory not chosen, terminating")
        exit()

    num_recordings = int(input("# of recordings: "))
    
    print("Press enter to start each recording\n")
    for i in range(num_recordings):
        input(f"Press Enter to start recording {i + 1}/{num_recordings}")
        audio_data = record_audio(SAMPLE_RATE, DURATION, device=5)
        timestamp = int(time.time() * 1000)
        filename = f"berry_{timestamp}.wav"
        filepath = os.path.join(save_directory, filename)
        write(filepath, SAMPLE_RATE, audio_data)
        print(f"saved recording {i + 1} as {filepath}\n")

    print("recording session complete, terminating")

if __name__ == "__main__":
    main()
