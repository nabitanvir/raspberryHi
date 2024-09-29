# After collecting a significant amount of wake word data, use this function to augment your files in order to introduce a more robust dataset.

import os
import random
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

POSITIVE_DIR = '/usr/src/app/berry/berry_voice_recognition/datasets/wake_word/positive'
NEGATIVE_DIR = '/usr/src/app/berry/berry_voice_recognition/datasets/wake_word/negative'
SAMPLE_RATE = 16000

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
    target = input('Augment positive or negative datasets? (p/n): ')
    tag = 'unset'
    if target == 'p':
        target_directory = POSITIVE_DIR
        tag = 'positive'
    elif target == 'n':
        target_directory = NEGATIVE_DIR
        tag = 'negative'
    else:
        print('No directory specified, terminating')
        exit()

    num_augmented_samples = int(input('how many augmented samples: '))
    if not num_augmented_samples:
        print('No number provided, terminating')

    print('Beginning augmentation!\n')
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f'created output directory: {target_directory}')

    target_files = []
    for f in os.listdir(target_directory):
        if f.endswith('.wav') and not f.startswith('augmented_'):
            target_files.append(os.path.join(target_directory, f))

    augmentations_per_file = int(np.ceil(num_augmented_samples / len(target_files)))

    file_iterator = tqdm(target_files, desc='processing files')
    augmented_count = 0
    with tqdm(total=num_augmented_samples, desc='augmenting samples\n', unit='sample') as pbar:
        for i, file in enumerate(target_files):
            audio, sr = librosa.load(file, sr=SAMPLE_RATE)
            base_filename = os.path.splitext(os.path.basename(file))[0]
            for j in range(augmentations_per_file):
                augmented_audio = audio.copy()

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

                if len(augmented_audio) > len(audio):
                    augmented_audio = augmented_audio[:len(audio)]
                else:
                    augmented_audio = np.pad(augmented_audio, (0, len(audio) - len(augmented_audio)), 'constant')

                max_val = np.max(np.abs(augmented_audio))
                if max_val > 1.0:
                    augmented_audio = augmented_audio / max_val


                output_file = os.path.join(target_directory, f"augmented_{tag}_{i}_{j}.wav")
                sf.write(output_file, augmented_audio, sr)
                augmented_count += 1
                pbar.update(1)

                if augmented_count >= num_augmented_samples:
                    break
                

            if augmented_count >= num_augmented_samples:
                break

        print(f"Generated {augmented_count} augmented {tag} samples, terminating")

if __name__ == "__main__":
    main()
