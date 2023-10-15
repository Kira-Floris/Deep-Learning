import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# sequential setup
# 1. return_sequences returns the output of each step, allowing to stack new RNN layers
model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(
    layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, activation="tanh")
    )
)
model.add(
    layers.Bidirectional(
        layers.LSTM(256, activation="tanh")
    )
)
model.add(
    layers.Dense(10)
)

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    verbose=2
)

model.evaluate(
    x_test, y_test,
    batch_size=32,
    verbose=2
)