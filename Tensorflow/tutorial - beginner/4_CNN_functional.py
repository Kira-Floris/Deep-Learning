import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# load data and change type and normalize
# no reshaping since we are using CNN that uses 2d instead of 1d arrays
# shape is (32,32,3) (height, width, RGB)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# functional model setup
# 1. layers.conv2d gets 32=output channel 
# 2. batchnormalization normalizes data for faster training, and also a regularization effect 
# 3. no activation function because of batchnormalization, add an activation function after batchnormalization
# 4. add kernel_regulazier to all conv2d and dense(64) layers
# 5. add dropout after dense(64)
# 6. adding dropout requires longer training given that there is a high dropout
def generate_model():
    inputs = keras.Input(shape=(32,32,3))
    x = layers.Conv2D(
        32,3, 
        padding="same", 
        kernel_regularizer=regularizers.l2(0.01)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(
        64,5, 
        padding="same", 
        kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    
    x = layers.Conv2D(
        128,3, 
        padding="same", 
        kernel_regularizer=regularizers.l2(0.01)
    )(x)
    
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    
    x = layers.Dense(
        64, 
        activation="relu",
        kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = generate_model()

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=32, epochs=150, verbose=2,)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
