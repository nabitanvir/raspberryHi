import os

# all files
POSITIVE_DIRECTORY = '/usr/src/app/berry/berry_voice_recognition/datasets/wake_word/positive'
NEGATIVE_DIRECTORY = '/usr/src/app/berry/berry_voice_recognition/datasets/wake_word/negative'
MODEL_SAVE_PATH = 'models/wake_word_model.h5'
SAMPLE_RATE = 16000
DURATION = 1.0
NUM_SAMPLES = SAMPLE_RATE * DURATION

# train_wake_word_model.py
WAKE_WORD_EPOCHS = 10
WAKE_WORD_BATCH_SIZE = 16
WAKE_WORD_VALIDATION_SPLIT = 0.2
WAKE_WORD_LEARNING_RATE = 0.001

# train_command_model.py
VOICE_COMMANDS = ['lights_on', 'lights_off', 'status']
DATA_DIR = 'datasets/commands'
MODEL_PATH = 'models/command_model.h5'

# record_wake_word.py
AUDIO_DEVICE_ID = 5

# utils.py

# transfer_train_wake_word_model.py
MODEL_PATH = 'models/wake_word_model.keras'

# transfer_train_command_model.py

# augment_wake_word_dataset.py

