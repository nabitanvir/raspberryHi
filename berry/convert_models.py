import tensorflow as tf

# This code converts the models into tflite models so it is less strenuous on the raspberry pi
def convert_to_tflite(model_path, output_path):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    printf("models converted")

convert_to_tflite('models/wake_word_model.h5', 'models/wake_word_model.tflite')
convert_to_tflite('models/command_model.h5', 'models/command_model.tflite')
