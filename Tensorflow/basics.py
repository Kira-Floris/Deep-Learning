"""
Definitions:
    tensor: multi dimensional array that can be run on GPU

"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import tensorflow as tf
import numpy as np

# initialization of tensors
# constants
x = tf.constant(4, shape=(1,1), dtype=tf.float32)
x_2d = tf.constant([[1,2,3],[1,2,3]]) 

ones = tf.ones((3,3))

zeros = tf.zeros((2,3))

identical_matrix = tf.eye(3)

random_normal = tf.random.normal((3,3), mean=0, stddev=1)

random_uniform = tf.random.uniform((4,3), minval=0, maxval=1)

range_n = tf.range(start=1, 
                   limit=10, 
                   delta=2 # skip 
                   )

