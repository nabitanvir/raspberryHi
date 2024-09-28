import os
import random
import librosa
import numpy as np
import soundfile as sf

NEGATIVE_DIR = 'datasets/wake_word/negative'
# temporary output dir to make sure audio files are being properly augmented
OUTPUT_DIR = 'datasets/wake_word/negative'

NUM_AUGMENTED_SAMPLES = 2700
SAMPLE_RATE = 16000

#def time_stretch_aug(audio, rate=1.0):
#    return librosa.effects.time_stretch(audio, rate)

def pitch_shift_aug(audio, sr, n_steps=0):
    shift = n_steps
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift)

def add_noise_aug(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def shift_audio_aug(audio, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
    return np.roll(audio, shift)

def change_volume_aug(audio, gain=1.0):
    return audio * gain

def main():
    print("beginning augmentation")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"created output directory: {OUTPUT_DIR}")

    negative_files = []
    for f in os.listdir(NEGATIVE_DIR):
        if f.endswith('.wav'):
            negative_files.append(os.path.join(NEGATIVE_DIR, f))

    num_augmented_samples = NUM_AUGMENTED_SAMPLES
    augmentations_per_file = int(np.ceil(num_augmented_samples / len(negative_files)))

    augmented_count = 0

    for i, file in enumerate(negative_files):
        audio, sr = librosa.load(file, sr=SAMPLE_RATE)
        for j in range(augmentations_per_file):
            augmented_audio = audio.copy()

            # i have no idea why this isnt working, debug for later model iterations
            #if random.random() < 0.5:
            #    rate = random.uniform(0.8, 1.2)
            #    augmented_audio = time_stretch_aug(augmented_audio, rate)
            if random.random() < 0.5:
                shift = random.randint(-2, 2)
                augmented_audio = pitch_shift_aug(augmented_audio, sr, n_steps=shift)
            if random.random() < 0.5:
                noise_factor = random.uniform(0.001, 0.005)
                augmented_audio = add_noise_aug(augmented_audio, noise_factor)
            if random.random() < 0.5:
                shift_max = 0.2
                augmented_audio = shift_audio_aug(augmented_audio, shift_max)
            if random.random() < 0.5:
                gain = random.uniform(0.7, 1.3)
                augmented_audio = change_volume_aug(augmented_audio, gain)

            # Make sure augmented audio has same length as original
            if len(augmented_audio) > len(audio):
                augmented_audio = augmented_audio[:len(audio)]
            else:
                augmented_audio = np.pad(augmented_audio, (0, len(audio) - len(augmented_audio)), 'constant')

            max_val = np.max(np.abs(augmented_audio))
            if max_val > 1.0:
                augmented_audio = augmented_audio / max_val

            output_file = os.path.join(OUTPUT_DIR, f"augmented_negative_{i}_{j}.wav")
            sf.write(output_file, augmented_audio, sr)
            augmented_count += 1

            if augmented_count >= NUM_AUGMENTED_SAMPLES:
                break

        if augmented_count >= NUM_AUGMENTED_SAMPLES:
            break

    print(f"Generated {augmented_count} augmented negative samples.")

if __name__ == "__main__":
    main()
