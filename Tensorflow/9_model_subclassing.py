"""
definitions:
1. subclassing allows you to create a model structure, similar to pytorch structure,
    allowing to easily reuse the class as many times as possible
"""
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
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding="same")
        self.bn = layers.BatchNormalization()
    
    # call method is same as forward in pytorch
    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x
    
"""
model subclassing for CNNBlock only
uncomment for testing and comment the codes following it
"""
# model = keras.Sequential(
#     [
#         CNNBlock(32),
#         CNNBlock(64),
#         CNNBlock(128),
#         layers.Flatten(),
#         layers.Dense(10)
#     ]
# )

# model.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"]
# )

# model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=2)
# print(model.summary())
# model.evaluate(x_test, y_test, batch_size=32, verbose=2)
    
class ResBlock(layers.Layer):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.cnn1 = CNNBlock(channels[0])
        self.cnn2 = CNNBlock(channels[1])
        self.cnn3 = CNNBlock(channels[2])
        self.pooling = layers.MaxPooling2D()
        self.identity_mapping = layers.Conv2D(channels[1], 1, padding="same")

    def call(self, input_tensor, training=False):
        x = self.cnn1(input_tensor, training=training)
        x = self.cnn2(x, training=training)
        x = self.cnn3(
            x + self.identity_mapping(input_tensor), training=training,
        )
        return self.pooling(x)

# keras.Model is same as layers.Layer but keras.Model has additional functionalities
# and best used in final model structure
class ResNet_Like(keras.Model):
    def __init__(self, num_classes=10):
        super(ResNet_Like, self).__init__()
        self.block1 = ResBlock([32,32,64])
        self.block2 = ResBlock([128,128,256])
        self.block3 = ResBlock([128,256,512])
        self.pool = layers.GlobalAveragePooling2D() # can be replaced by layers.Flatten()
        self.classifier = layers.Dense(num_classes)

    def call(self, input_tensor, training=False):
        x = self.block1(input_tensor, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.pool(x)
        return self.classifier(x)
    
    # this allows to see information of output shape, instead of getting multiple as answer in summary
    def get_summary(self):
        x = keras.Input(shape=(28,28,1))
        return keras.Model(inputs=[x], outputs=self.call(x))

model = ResNet_Like(num_classes=10)
   
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=2)
print(model.get_summary().summary())
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
