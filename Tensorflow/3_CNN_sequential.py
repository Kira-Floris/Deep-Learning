import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# load data and change type and normalize
# no reshaping since we are using CNN that uses 2d instead of 1d arrays
# shape is (32,32,3) (height, width, RGB)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# sequential model setup
# 1. layers.conv2d gets 32=output channel 
model = keras.Sequential(
    [
        keras.Input(shape=(32,32,3)),
        layers.Conv2D(32,(3,3), padding="valid", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64,(3,3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128,(3,3),activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10),
    ]
)

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2,)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
