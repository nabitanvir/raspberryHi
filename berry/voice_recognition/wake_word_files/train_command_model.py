import os, librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

COMMANDS = ['lights_on', 'lights_off', 'status']
DATA_DIR = 'datasets/commands'
MODEL_PATH = 'models/command_model.h5'

# We preprocess data into MFCCs of dimensions 32x32x1, which is the equivalent of a grayscale image
def load_command_data(commands, data_dir):
    X = []
    y = []
    for idx, command in enumerate(commands):
        command_dir = os.path.join(data_dir, command)
        for file in os.listdir(command_dir):
            filepath = os.path.join(command_dir, file)
            audio, sr = librosa.load(filepath, sr=16000)
            mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
            mfcc = np.resize(mfcc, (32, 32))
            X.append(mfcc)
            y.append(idx)
    X = np.array(X)
    X = X[..., np.newaxis]
    y = np.array(y)
    return X, y

# Create the model, this specific NN architecture was used to make for a lightweight model that can easily run on a raspberry pi
def create_command_model(input_shape, num_commands):
    model = models.sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_commands, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("loading command data")
    X_train, y_train = load_command_data(COMMANDS, DATA_DIR)
    print("data loaded, training model!")
    input_shape = (32, 32, 1)
    num_commands = len(COMMANDS)
    model = create_command_model(input_shape, num_commands)
    model.fit(X_train, y_train, epochs=15, batch_size=16, validation_split=0.2)
    model.save(MODEL_PATH)
    print("model completed, saved!")

if __name__ == "__main__":
    main()
