import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models

POSITIVE_DIR = 'datasets/wake_word/positive'
NEGATIVE_DIR = 'datasets/wake_word/negative'
MODEL_PATH = 'models/wake_word_model.h5'

def load_wake_word_data(positive_dir, negative_dir):
    X = []
    y = []
    for file in os.listdir(positive_dir):
        filepath = os.path.join(positive_dir, file)
        audio, sr = librosa.load(filepath, sr=16000)
        mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
        mfcc = np.resize(mfcc (32,32,))
        X.append(mfcc)
        y.append(1)
    for file in os.listdir(negative_dir):
        filepath = os.path.join(negative_dir, file)
        audio, sr = librosa.load(filepath, sr=16000)
        mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
        mfcc = np.resize(mfcc, (32,32))
        X.append(mfcc)
        y.append(0)
    X = np.array(X)
    X = X[..., np.newaxis]
    y = np.array(y)
    return X, y

# We use a relatively small amount of layers to make sure the model isn't too complex for the raspberry pi
def create_wake_word_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(8, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("Loading wake word data")
    X_train, y_train = load_wake_word_data(POSITIVE_DIR, NEGATIVE_DIR)
    print("Data loaded, training model!")
    input_shape = (32, 32, 1)
    model = create_wake_word_model(input_shape)
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)
    model.save(MODEL_SAVE_PATH)
    print(f"Wake word model saved at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
