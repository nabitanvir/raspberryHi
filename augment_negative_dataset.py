import os
import random
import librosa
import numpy as np

NEGATIVE_DIR = 'datasets/wake_word/negative'
# temporary output dir to make sure audio files are being properly augmented
OUTPUT_DIR = 'datasets/wake_word/augmentation_test'

def time_stretch(audio, rate=1.0):
    return librosa.effects.time_stretch(audio, rate)

def pitch_shift(audio, sr, n_steps=2):
    return librosa.effects.pitch_shift(audio, sr, n_steps=n_steps)

def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def shift_audio(audio, shift_max=0.2, sr=16000):
    shift = np.random.randint(sr * shift_max)
    return np.roll(audio,shift)

def change_volume(audio, gain=0.2):
    return audio * gain

def main():
    negative_files = []
    for f in os.listdir(NEGATIVE_DIR):
        if f.endswith('.wav'):
            negative_files.append(os.path.join(NEGATIVE_DIR, f))

    num_augmented_samples = 500
    augmentations_needed = num_augmented_samples - len(negative_files)
    augmentations_per_file = augmentations_needed // len(negative_files) + 1

    augmented_files = []
    for i, file in enumerate(negative_files):
        audio, sr = librosa.load(file, sr=None)
        for j in range(augmentations_per_file):
            augmented_audio = audio
            if random.random() > 0.5:
                augmented_audio = time_stretch(augmented_audio, random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                augmented_audio = pitch_shift(augmented_audio, sr, random.randint(-2, 2))
            if random.random() > 0.5:
                augmented_audio = add_noise(augmented_audio, noise_factor=random.uniform(0.002, 0.01))
            if random.random() > 0.5:
                augmented_audio = shift_audio(augmented_audio, shift_max=0.2, sr=sr)
            if random.random() > 0.5:
                augmented_audio = change_volume(augmented_audio, gain=random.randint(0.7, 1.3))

            output_file = os.path.join(output_dir, f"augmented_negative_{i}_{j}.wav")
            librosa.output.write_wav(output_file, aug_audio, sr)
            augmented_files.append(output_file)

            if len(augmented_files) >= augmentations_needed:
                break

        if len(augmented_files) >= augmentations_needed:
            break

    print(f"Generated {len(augmented_files)} augmented negative samples.")
    return 1

if __name__ == "__main__":
    main()
