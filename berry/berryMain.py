# berryMain.py

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd

# Constants
SAMPLE_RATE = 16000  # 16 kHz
DURATION = 1         # 1 second
NUM_SAMPLES = SAMPLE_RATE * DURATION
MODEL_PATH = 'models/wake_word_model.keras'
THRESHOLD = 0.5      # Adjust based on your model's performance

# Define the custom YamnetEmbeddingLayer used in your model
class YamnetEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, yamnet_model_handle='https://tfhub.dev/google/yamnet/1', **kwargs):
        super(YamnetEmbeddingLayer, self).__init__(**kwargs)
        self.yamnet_model_handle = yamnet_model_handle
        # Defer loading the YAMNet model to the build method
        self.yamnet_model = None

    def build(self, input_shape):
        # Load the YAMNet model during the build phase
        self.yamnet_model = hub.load(self.yamnet_model_handle)
        super(YamnetEmbeddingLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, NUM_SAMPLES)
        def extract_embeddings(waveform):
            # YAMNet expects input of shape (num_samples,)
            _, embeddings, _ = self.yamnet_model(waveform)
            # Average over time frames to get a fixed-size embedding
            return tf.reduce_mean(embeddings, axis=0)

        embeddings = tf.map_fn(
            lambda x: extract_embeddings(x),
            inputs,
            fn_output_signature=tf.float32
        )
        return embeddings

    def get_config(self):
        config = super(YamnetEmbeddingLayer, self).get_config()
        config.update({
            'yamnet_model_handle': self.yamnet_model_handle,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Load your trained model with the custom layer
print("Loading the wake word detection model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'YamnetEmbeddingLayer': YamnetEmbeddingLayer}
)
print("Model loaded successfully.")

def get_audio_input():
    """Record 1 second of audio from the microphone and preprocess it."""
    print("Please say the wake word...")
    # Record audio
    audio = sd.rec(int(NUM_SAMPLES), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished

    # Convert to mono and flatten
    audio = np.squeeze(audio)

    # Ensure the audio is exactly NUM_SAMPLES long
    if len(audio) > NUM_SAMPLES:
        audio = audio[:NUM_SAMPLES]
    elif len(audio) < NUM_SAMPLES:
        padding = NUM_SAMPLES - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')

    return audio

def predict_wake_word(audio_data):
    """Predict whether the wake word is present in the audio data."""
    # Expand dimensions to match model input shape
    audio_data = np.expand_dims(audio_data, axis=0)  # Shape: (1, NUM_SAMPLES)
    # Make prediction
    probability = model.predict(audio_data)[0][0]
    return probability

def main():
    print("Starting wake word detection...")
    while True:
        audio_data = get_audio_input()
        probability = predict_wake_word(audio_data)
        if probability > THRESHOLD:
            print("hello! :o")
        else:
            print("lonely :(")

if __name__ == "__main__":
    main()

