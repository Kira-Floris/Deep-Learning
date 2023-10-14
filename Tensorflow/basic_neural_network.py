import os
os.environ["TFF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# loading dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
transformation:
1. reshape from multi dimension to single dimension [[1,2,3],[1,2,3]] to [1,2,3,1,2,3]
2. change datatype from float64 to float32 for faster processing, float64 heavy
3. normalize from 255.0 to 1

information:
1. images here are 28 by 28, which is why in reshape we use (-1, 28*28) or (-1, 784)
"""
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

"""
model initialization
1. last layer should contain the number of classification categories
"""

"""
Sequential API
1. use Sequential API (not very flexible, one input one output)
2. change loss from_logits argument to True
"""
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10),
    ]
)

# second alternative to creating a model structure
model_alt = keras.Sequential()
model_alt.add(keras.Input(shape=(28*28)))
model_alt.add(layers.Dense(512, activation="relu"))
model_alt.add(layers.Dense(256, activation="relu"))
model_alt.add(layers.Dense(10))

"""
Functional API
1. more flexible, multiple inputs multiple outputs
2. add the previous layer to the current layer as input
3. in outputs, define activation function
4. change loss from_logits argument to False
"""
inputs = keras.Input(shape=(28*28))
x = layers.Dense(512, activation="relu")(inputs)
x = layers.Dense(256, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)


"""
model summary:
1. if we add keras.Input(shape=(28*28)) in the layers we can get model summary
2. if not added, we can get the summary after FITTING
"""
print(model.summary())

"""
model compiler
1. configuring loss function, optimizer and metrics score
"""
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

"""
model training
"""
model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=5,
    verbose=2
)

"""
model evaluation
"""
model.evaluate(x_test, y_test, batch_size=32, verbose=2)

