import config
import utils

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPool2D, Lambda

def build_siamese_network(inputShape, embeddingDim=48):
    inputs = Input(inputShape)

    x = Conv2D(filters=64, strides=(2, 2), padding='same', activation='relu')(inputs)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.3)(x)

    x = Conv2D(filters=64, strides=(2, 2), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.3)(x)

    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(embeddingDim)(pooledOutput)

    model = Model(inputs, outputs)

    return model

def create_siamese_network():
    base_network = build_siamese_network(config.IMG_SHAPE, config.EMBEDDING_SIZE)
    
    input_a = Input(shape=config.IMG_SHAPE)
    input_b = Input(shape=config.IMG_SHAPE)

    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)

    distance = Lambda(utils.euclidean_distance)([embedding_a, embedding_b])
    
    model = Model(inputs=[input_a, input_b], outputs=distance)
    return model
    
def prepare_dataset(dataset):
    image1_array = np.array([img1 for img1, _, _ in dataset], dtype=np.float32)
    image2_array = np.array([img2 for _, img2, _ in dataset], dtype=np.float32)
    labels_array = np.array([label for _, _, label in dataset], dtype=np.float32)
    
    labels_array = labels_array.reshape(-1, 1)

    return [image1_array, image2_array], labels_array
    
def train_siamese_network():
    print("[INFO] Creating Siamese neural network...")
    model = create_siamese_network(config.IMG_SHAPE, config.EMBEDDING_SIZE)
    print("[INFO] Siamese neural network created.")

    print("[INFO] Generating dataset for training...")
    dataset = utils.create_pairs(config.POSITIVE_DATASET_PATH, config.NEGATIVE_DATASET_PATH)
    print("[INFO] Dataset generated.")

    print("[INFO] Preparing dataset...")
    (image1_array, image2_array), labels_array = prepare_dataset(dataset)
    print("[INFO] Dataset prepared.")

    print("[INFO] Starting training...")
    model.compile(optimizer='adam', loss=config.contrastive_loss)
    model.fit([image1_array, image2_array], labels_array,
              batch_size=config.BATCH_SIZE,
              epochs=config.EPOCHS)
    print("[INFO] Training completed.")

    model.save(config.MODEL_OUTPUT)
    print(f"[SUCCESS] Model saved to {config.MODEL_OUTPUT}")

    