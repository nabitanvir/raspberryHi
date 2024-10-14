import config
import utils

import os
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras import models, layers

# Preprocess audio to match training
def preprocess_audio(audio):
    # Ensure audio is exactly 1 second
    audio = librosa.util.fix_length(audio, size=NUM_SAMPLES)
    
    # Compute MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    
    # Truncate or pad MFCCs to have 32 frames
    max_length = 32
    if mfcc.shape[1] > max_length:
        mfcc = mfcc[:, :max_length]
    elif mfcc.shape[1] < max_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
    
    # Expand dimensions to match model input
    mfcc = mfcc[..., np.newaxis]  # Shape: (13, 32, 1)
    return mfcc

# Load the trained model
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
    
    model = models.load_model(MODEL_PATH)
    return model

# Record audio from microphone
def record_audio():
    print("Press Enter to record...")
    input()  # Wait for user to press Enter
    print("Recording...")
    
    # Record audio
    audio = sd.rec(int(NUM_SAMPLES), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    audio = np.squeeze(audio)  # Shape: (16000,)
    print("Recording complete.")
    return audio

# Predict wake word
def predict(model, audio):
    mfcc = preprocess_audio(audio)
    mfcc = np.expand_dims(mfcc, axis=0)  # Shape: (1, 13, 32, 1)
    
    # Make prediction
    prediction = model.predict(mfcc)[0][0]
    return prediction

# Main loop
def main():
    print("Loading the wake word detection model...")
    model = load_model()
    print("Model loaded successfully.\n")
    
    while True:
        try:
            audio = record_audio()
            probability = predict(model, audio)
            
            if probability > THRESHOLD:
                print("hello! :o\n")
            else:
                print("lonely :(\n")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}\n")

if __name__ == "__main__":
    main()

