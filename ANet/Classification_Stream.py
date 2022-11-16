import tensorflow as tf


def classification_stream():
    inputs = tf.keras.Input(shape=(None, None, 3))
    layer1 = tf.keras.layers.Dense(2048)(inputs)
    layer2 = tf.keras.layers.ReLU()(layer1)
    layer3 = tf.keras.layers.Dropout(0.5)(layer2)
    layer4 = tf.keras.layers.Dense(2048)(layer3)
    layer5 = tf.keras.layers.ReLU()(layer4)
    layer6 = tf.keras.layers.Dropout(0.5)(layer5)
    layer7 = tf.keras.layers.Dense(2)(layer6)
    layer8 = tf.keras.layers.Softmax()(layer7)
    outputs = layer8
    return tf.keras.Model(inputs, outputs, name="Classification_Stream")
