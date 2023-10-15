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

"""
1. how to save and load model weights
    requires to load the model with the exact model structure when created
2. save and load entire model (serializing model)
    - save weights
    - model architecture
    - training configuration (model.compile() information)
    - optimizer and states
    - this gives you the ability to continue training where the model left off training
"""
model1 = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(64, activation="relu"),
        layers.Dense(10)
    ]
)

model1.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"]
)

model1.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)
model1.evaluate(x_test, y_test, batch_size=32, verbose=2)

# 1. save and load model weights
# 1.1 save model weight
model1.save_weights("models_folder/save_model/")
# 1.2 load model weights
model1.load_weights("models_folder/save_model/")
print(model1.summary())

# 2. saving complete model
# 2.1 saving model weight
model1.save("models_folder/complete_model/")
# 2.1 load complete model
model = keras.models.load_model("models_folder/complete_model/")
print(model.summary())
