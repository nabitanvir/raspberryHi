# My NN architecture for training the wake word model, I use a CNN model with 2D MFCC inputs

import os, librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

POSITIVE_DIR = 'datasets/wake_word/positive'
NEGATIVE_DIR = 'datasets/wake_word/negative'
MODEL_PATH = 'models/wake_word_model.h5'

# This function preprocesses our audio data for training by converting all our training data into MFCCs of size 32x32x1, which is
# the equivalent of a grayscale image (hence the 1 in the channels dimension)
def load_wake_word_data(positive_dir, negative_dir):
    X = []
    y = []
    max_length = 32
    for file in os.listdir(positive_dir):
        filepath = os.path.join(positive_dir, file)
        audio, sr = librosa.load(filepath, sr=16000)
        
        # ensure audio length is consistent
        audio = librosa.util.fix_length(audio, size=16000)
        
        mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
        #mfcc = np.resize(mfcc (32,32,))
        
        #ensure mfcc length is consistent
        mfcc = mfcc[:, :max_length]
        if mfcc.shape[1] < max_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')

        X.append(mfcc)
        y.append(1)

    for file in os.listdir(negative_dir):
        filepath = os.path.join(negative_dir, file)
        audio, sr = librosa.load(filepath, sr=16000)
        
        audio = librosa.util.fix_length(audio, size=16000)
        
        mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
        #mfcc = np.resize(mfcc, (32,32))
        if mfcc.shape[1] < max_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')

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

    optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main function that trains the model
def main():
    print("Loading wake word data")
    X, y = load_wake_word_data(POSITIVE_DIR, NEGATIVE_DIR)
    print("Data loaded, training model!")

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    validation_split = 0.2
    num_validation_samples = int(validation_split * len(X))
    X_train = X[:-num_validation_samples]
    y_train = y[:-num_validation_samples]
    X_val = X[-num_validation_samples:]
    y_val = y[-num_validation_samples:]

    input_shape = X_train.shape[1:]
    model = create_wake_word_model(input_shape)
    model.summary()
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val), verbose=1)
    model.save(MODEL_PATH)
    print(f"Wake word model saved at {MODEL_PATH}")

if __name__ == "__main__":
    main()
