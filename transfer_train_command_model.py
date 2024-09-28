# In this training code, we train command recognition using a pretrained speech embedding model
# https://www.kaggle.com/models/google/speech-embedding/tensorFlow1/speech-embedding/1?tfhub-redirect=true

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

COMMANDS = ['lights_on', 'lights_off', 'status']
DATA_DIR = 'datasets/commands'
MODEL_PATH = 'models/command_model.h5'

SAMPLE_RATE = 16000
DURATION = 1

def load_wake_word_data(positive_dir, negative_dir):
    positive_files = []
    negative_files = []

    for f in os.listdir(positive_dir):
        if f.endswith('.wav'):
            positive_files.append(os.path.join(positive_dir, f))

    for f in os.listdir(negative_dir):
        if f.endswith('.wav'):
            negative_files.append(os.path.join(negative_dir, f))

    positive_waveforms = load_audio_files(positive_files)
    negative_waveforms = load_audio_files(negative_files)

    X = tf.concat([positive_waveforms, negative_waveforms], axis=0)
    y = tf.concat([tf.ones(len(positive_waveforms)), tf.zeros(len(negative_waveforms))], axis=0)

    return X.numpy(), y.numpy()

def load_command_data(commands, data_dir):
    X = []
    y = []
    label_map = {}
    for i, command in enumerate(commands):
        label_map[command] = i

    for command in commands:
        command_dir = os.path.join(data_dir, command)
        for f in os.listdir(command_dir):
            if f.endswith('.wav'):
                command_files.append(os.path.join(command_dir, f))
        waveforms = load_audio_files(command_files)
        labels = [label_map[command]] * len(waveforms)
        X.append(waveforms)
        y.extend(labels)
    X = tf.concat(X, axis=0)
    y = np.array(y)
    return X.numpy(), y

def create_command_model(num_commands):
    base_model = hub.KerasLayer('https://tfhub.dev/google/speech_embedding/1', trainable=False)
    input_waveform = tf.keras.Input(shape=(NUM_SAMPLES,), dtype=tf.float32)

    embeddings = base_model(tf.expand_dims(input_waveform, axis=-1))
    embeddings = tf.squeeze(embeddings, axis=1)

    x = tf.keras.layers.Dense(256, activation='relu')(embeddings)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(num_commands, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_waveform, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("PRETRAIN VER: Loading command data")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(16).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(16).prefetch(tf.data.AUTOTUNE)

    num_commands = len(COMMANDS)
    model = create_command_model(num_commands)
    model.summary()

    print("PRETRAIN VER: Training command recognition model")
    model.fit(train_dataset, validation_data=val_dataset, epochs=15)

    model.save(MODEL_SAVE_PATH)
    printf("PRETRAIN VER: command model trained!")

if __name__ == "__main__":
    main()
