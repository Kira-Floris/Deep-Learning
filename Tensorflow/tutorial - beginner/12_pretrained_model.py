import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

"""
pretrained model, model I trained or found on github
1. NB: the model used here was trained and saved using 11_saving_model.py codes
2. for fine tuning, freeze the training by adding model.trainable=False after loading model
    freezing makes the model run faster
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1,28*28).astype("float32") / 255.0

model = keras.models.load_model("models_folder/complete_model/")

# freezing method 1
model.trainable = False

# freezing method 2, this can also be used to select some layers to freeze with model.layers[:1]
for layer in model.layers:
    assert layer.trainable == False
    layer.trainable = False

# changing the last layer to have a different number of output/classes
new_n_classes = 10
base_inputs = model.layers[0].input
base_outputs = model.layers[-2].output
final_outputs = layers.Dense(new_n_classes, name="dense_out")(base_outputs)

new_model = keras.Model(inputs=base_inputs, outputs=final_outputs)
print(new_model.summary())

new_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
new_model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=2)
new_model.evaluate(x_test, y_test, batch_size=32, verbose=2)

"""
pretrained keras models
1. we use some random data
"""
x = tf.random.normal(shape=(5,299,299,3))
y = tf.constant([0,1,2,3,4])

pretrainedk_model = keras.applications.InceptionV3(include_top=True)
# print(pretrainedk_model.summary())

new_n_classes = 5
base_inputs = pretrainedk_model.layers[0].input
base_outputs = pretrainedk_model.layers[-2].output
final_outputs = layers.Dense(new_n_classes)(base_outputs)
new_pretrainedk_model = keras.Model(inputs=base_inputs, outputs=final_outputs) 
# print(new_pretrainedk_model.summary())

new_pretrainedk_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
new_pretrainedk_model.fit(x,y,epochs=2, batch_size=32, verbose=2)

"""
tensorflow hub
"""
x = tf.random.normal(shape=(5,299,299,3))
y = tf.constant([0,1,2,3,4])

url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"

base_model = hub.KerasLayer(url, input_shape=(299,299,3))
base_model.trainable = False
hub_model = keras.Sequential(
    [
        base_model,
        layers.Dense(129,activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(5),
    ]
)

hub_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

hub_model.fit(x,y,batch_size=32,epochs=10,verbose=2)