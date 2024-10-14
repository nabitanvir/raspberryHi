import os

IMG_SHAPE = (224, 224, 3)
EMBEDDING_SIZE = 64

BATCH_SIZE = 64
EPOCHS = 100

BASE_PATH = 'berry/models/'
MODEL_OUTPUT = os.path.join(BASE_PATH, 'siamese_model')
PLOT_OUTPUT = os.path.join(BASE_PATH, 'training_graph.png')