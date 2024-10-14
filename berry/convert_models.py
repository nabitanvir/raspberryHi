import os
import tensorflow as tf

def convert_to_tflite(model_path, output_path):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

def main():
    print("Converting models to TFlite models...")
    if os.path.isfile('models/wake_word_model.h5'):
        print("Converting wake work model...")
        convert_to_tflite('models/wake_word_model.h5', 'models/wake_word_model.tflite')
    if os.path.isfile('models/command_model.h5'):
        print("Converting command model...")
        convert_to_tflite('models/command_model.h5', 'models/command_model.tflite')
    if os.path.isfile('models/face_model.h5'):
        print("Converting face model...")
        convert_to_tflite('models/face_model.h5', 'models/command_model.tflite')
    print("Completed conversion!")
