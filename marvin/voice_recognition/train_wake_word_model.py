import os

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

import config
import utils

def load_wake_word_data(positive_dir, negative_dir):
    """
    Preprocesses audio data for wake word detection by converting audio files into Mel-Frequency Cepstral Coefficients (MFCCs).
    The output MFCCs are resized to a consistent shape of 32x32x1, resembling a grayscale image.

    Parameters:
    positive_dir (str): Directory containing audio files with positive examples (wake word present).
    negative_dir (str): Directory containing audio files with negative examples (wake word absent).

    Returns:
    X (numpy.ndarray): Array of MFCCs with shape (number of samples, 32, 32, 1), representing preprocessed audio data.
    y (numpy.ndarray): Array of labels with shape (number of samples,), where 1 indicates a positive sample and 0 indicates a negative sample.
    """
    X = []
    y = []
    max_length = 32
    for file in os.listdir(positive_dir):
        filepath = os.path.join(positive_dir, file)
        audio, sr = librosa.load(filepath, sr=16000)
        audio = librosa.util.fix_length(audio, size=16000)
        mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
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
        if mfcc.shape[1] < max_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')

        X.append(mfcc)
        y.append(0)
        
    X = np.array(X)
    X = X[..., np.newaxis]
    y = np.array(y)
    return X, y

def create_wake_word_model(input_shape, learning_rate):
    """
    Creates a convolutional neural network (CNN) model for wake word detection. 
    The model is designed to be small and efficient, suitable for deployment on 
    resource-constrained devices like a Raspberry Pi.

    Parameters:
    input_shape (tuple): Shape of the input data, typically (32, 32, 1), representing 
                         the dimensions of the preprocessed MFCCs.

    Returns:
    model (tensorflow.keras.Model): Compiled CNN model ready for training.
    """
    model = models.Sequential([
        layers.Conv2D(8, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_wake_word_model(X, y, validation_split=0.2, epochs=10, batch_size=16, learning_rate=0.05):
    """
    Trains the wake word detection model using the provided data.
    
    Parameters:
    X (numpy.ndarray): Array of preprocessed audio data (MFCCs).
    y (numpy.ndarray): Array of corresponding labels (1 for positive, 0 for negative).
    validation_split (float): Fraction of the data to be used for validation (default is 0.2).
    epochs (int): Number of epochs to train the model (default is 10).
    batch_size (int): Number of samples per gradient update (default is 16).
    
    Returns:
    model (tensorflow.keras.Model): The trained model.
    """
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    num_validation_samples = int(validation_split * len(X))
    X_train = X[:-num_validation_samples]
    y_train = y[:-num_validation_samples]
    X_val = X[-num_validation_samples:]
    y_val = y[-num_validation_samples:]

    input_shape = X_train.shape[1:]

    model = create_wake_word_model(input_shape, learning_rate)
    model.summary()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)
    
    return model

def main():
    """
    Main function that loads the data, trains the wake word model, and saves it.
    """
    print("[INFO] Loading wake word data...")
    X, y = load_wake_word_data(config.POSITIVE_DIRECTORY, config.NEGATIVE_DIRECTORY)
    print("[INFO] Data loaded!")
    
    print("[INFO] Training wake word model...")
    model = train_wake_word_model(X, y, validation_split=config.WAKE_WORD_VALIDATION_SPLIT,
                                  epochs=config.WAKE_WORD_EPOCHS,
                                  batch_size=config.WAKE_WORD_BATCH_SIZE,
                                  learning_rate=config.WAKE_WORD_LEARNING_RATE)
    print("[INFO] Training wake word model trained!")

    model.save(config.MODEL_SAVE_PATH)
    print(f"[INFO] Wake word model saved at {config.MODEL_SAVE_PATH}")