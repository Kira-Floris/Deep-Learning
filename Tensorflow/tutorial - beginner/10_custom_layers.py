import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

class CustomDense(layers.Layer):
    def __init__(self, units):
        super(CustomDense, self).__init__()
        self.units = units
        

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.units, ),
            initializer="zeros",
            trainable=True
        )
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
class CustomRelu(layers.Layer):
    def __init__(self):
        super(CustomRelu, self).__init__()

    def call(self, x):
        return tf.math.maximum(x, 0)   

class CustomModel(keras.Model):
    def __init__(self, num_classes=10):
        super(CustomModel, self).__init__()

        # custom layers
        self.dense1 = CustomDense(64)
        self.dense2 = CustomDense(num_classes)
        self.relu = CustomRelu()

        # normal layers
        # self.dense1 = layers.Dense(64)
        # self.dense2 = layers.Dense(num_classes)

    def call(self, input_tensor):
        x = self.relu(self.dense1(input_tensor))
        return self.dense2(x)
    
model = CustomModel()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)
print(model.summary())
model.evaluate(x_test, y_test, batch_size=32, verbose=2)