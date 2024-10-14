# siamese network approach

import berry.face_recognition.config as config
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPool2D

def siamese_network(inputShape, embeddingDim=48):
    inputs = Input(inputShape)

    x = Conv2D(filters=64, stride=(2, 2), padding='same', activation='relu')(inputs)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.3)(x)

    x = Conv2D(filters=64, stride=(2, 2), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.3)(x)

    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(embeddingDim)(pooledOutput)

    model = Model(inputs, outputs)

    return model

def train_siamese_network():
    print("[INFO] Loading face dataset...")

def main():
    siamese_network(config.IMG_SHAPE, config.EMBEDDING_SIZE)

    