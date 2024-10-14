# siamese network approach

import siamese_config
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

def main():
    siamese_network(siamese_config.IMG_SHAPE, siamese_config.EMBEDDING_SIZE)
    siamese_model.summary()

    