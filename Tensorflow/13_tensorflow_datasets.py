import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(ds_train, ds_test), ds_info = tfds.load(
    "mnist", # dataset name in tensorflow catalog
    split=["train", "test"], # if dataset has validation data, make it ["train","val","test"]
    shuffle_files=True, 
    as_supervised=True, # return tuple of (img, label) instead of a dictionary
    with_info=True
)

print(ds_info)