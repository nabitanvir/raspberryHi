import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import librosa
import os

DATASET_DIR = "dataset/"
TRAIN_DIR = os.path.join(DATASET_DIR, "train/")
VALIDATION_DIR = os.path.join(DATASET_DIR, "validation/")

def load_audio_data(directory, labels):
    X = []
    y_wake_word = []
    y_command = []
    for label in labels:
        folder = os.path.join(directory, label)
        for file in os.listdir(folder):
            filepath = os.path.join(folder, file)
            audio_data, sample_ratee = librosa.load(filepath, sr=16000)
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            X.append(mfcc)
            if label == "berry":
                y_wake_word.append(1)
                y_command.append(-1)
            else:
                y_wake_word.append(0)
                y_command.append(labels.index(label))
        return np.array(X), np.array(y_wake_word), np.array(y_command)

# Due to lack of processing power, we will use transfer learning with a pretrained MobileNet model
# MobileNet is ideal here because it uses depthwise separable convolution, which significantly reduces needed computations (Thanks Andrew Ng!)
def get_pretrained_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
    # We freeze the base model so we can add custom layers that introduce bias to our own voice
    base_model.trainable = False
    return base_model

def build_model(base_model, num_classes):
    inputs = Input(shape=(64, 64, 1))
    
    # We stack the input three times so we can match the 3 channel requirement for MobileNet
    x = layers.Concatenate()([inputs, inputs, inputs])
    x = base_model(x)

    x = layers.GlobalAveragePooling2D()(x)

    # For wake word detection we use sigmoid
    wake_word_branch = layers.Dense(128, activation='relu')(x)
    wake_word_output = layers.Dense(1, activation='sigmoid', name='wake_word')(wake_word_branch)

    # For command recognition, we use a softmax
    command_branch = layers.Dense(128, activation='relu')(x)
    command_output = layers.Dense(num_classes, activation='softmax', name='command')(command_branch)

    model = models.Model(inputs=inputs, outputs=[wake_word_output, command_output])

    # We check for two separate losses, one for our wake word and another for our commands
    model.compile(optimizer='adam',
                  loss={'wake_word': 'binary_crossentropy', 'command': 'sparse_categorical_crossentropy'},
                  metrics={'wake_word': 'accuracy', 'command': 'accuracy'})

    return model

def main():
    # This is where you can add your own custom commands or wake word, just make sure to have a substantial amount of clean and labeled data.
    labels = ["berry", "lights_on", "lights_off"]
    
    print("Loading datasets")
    X_train, y_train_wake_word, y_train_command = load_audio_data(TRAIN_DIR, labels)
    X_validation, y_validation_wake_word, y_validation_command = load_audio_data(VALIDATION_DIR, labels)

    # For any confusion why we do this, its because CNN models tend to take a (height, width, channels), where channels are for RGB (3 channels)
    # MFCC are 2D arrays, meaning they have no third channel, so we add a third channel so the CNN can take this input, this is essentially the same as a greyscale image (1 channel)
    X_train = np.expand_dims(X_train, -1)
    X_validation = np.expand_dims(X_validation, -1)

    base_model = get_pretrained_model()
    # now we add our own layers to the pretrained model
    model = build_model(base_model, num_classes=len(labels))

    print("Training model!")
    # NOTE: The amount of epochs here will need to be changed depending on many aspects (dataset size, model complexity, learning rate, etc)
    # I recommend training a couple of times and checking your loss, make sure not to overfit though!
    model.fit(X_train, {'wake_word': y_train_wake_word, 'command': y_train_command},
              validation_data=(X_validation, {'wake_word': y_val_wake_word, 'command': y_val_command}),
              epochs=10, batch_size=32)

    model.save('berryAudioModel.h5')
    print("Model saved as berryAudioModel.h5!")

if __name__ == "__main__":
    main()

