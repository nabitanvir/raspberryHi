import config

import os
import tensorflow as tf

# Convert our .h5 into .tflite
def convert_to_tflite(model_path, output_path):
    print(f"Converting {os.path.basename(model_path)} into tflite model")
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Converted {os.path.basename(model_path)} into tflite model")

def load_model(file_path):
    return None