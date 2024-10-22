import os

# siamese_network.py
IMG_SHAPE = (224, 224, 3)
EMBEDDING_SIZE = 64

BATCH_SIZE = 64
EPOCHS = 100

MODEL_SAVE_PATH = 'berry/models/'
PLOT_SAVE_PATH = 'berry/plots/'
MODEL_OUTPUT = os.path.join(MODEL_SAVE_PATH, 'siamese_model')
PLOT_OUTPUT = os.path.join(PLOT_SAVE_PATH, 'siamese_training_graph.png')

# utils.py
POSITIVE_DATASET_PATH = 'berry/face_recognition/dataset/positive'
NEGATIVE_DATASET_PATH = 'berry/face_recognition/dataset/negative'
PAIRS_PER_PERSON = 30