"""
Definitions:
    tensor: multi dimensional array that can be run on GPU

"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import tensorflow as tf
import numpy as np

"""
setting up GPU and
stop tensorflow from allocating all memory to the process
"""
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


"""
initialization of tensors
"""
# constants
x = tf.constant(4, shape=(1,1), dtype=tf.float32)
x_2d = tf.constant([[1,2,3],[1,2,3]]) 

# ones, [1,1,1]
ones = tf.ones((3,3))

# zeros, [0,0,0]
zeros = tf.zeros((2,3))

# identical matrix, [[1,0,0],[0,1,0],[0,0,1]]
identical_matrix = tf.eye(3)

# random matrices
random_normal = tf.random.normal((3,3), mean=0, stddev=1)
random_uniform = tf.random.uniform((4,3), minval=0, maxval=1)

# range matrices
range_n = tf.range(start=1, 
                   limit=10, 
                   delta=2 # skip 
                   )

# cast is used for changing data type
range_cast_type = tf.cast(
    range_n,
    dtype=tf.float64
)

"""
mathematical operations
"""
x = tf.constant([1,2,3])
y = tf.constant([9,8,7])
 
# basic operations
z = tf.add(x,y) # or x + y
z = tf.subtract(x,y) # or x - y
z = tf.divide(x,y) # or x / y
z = tf.multiply(x,y) # x * y

# dot product
dot_product = tf.tensordot(x,y,axes=1) # or tf.reduce_sum(x*y, axes=0)

# constant power
z = x ** 5

# matrix multiplication
x = tf.random.normal((2,3))
y = tf.random.normal((3,4))
matrix_multiply = tf.matmul(x,y) # or x @ y

"""
indexing, i
"""
x = tf.constant([0,1,2,3,4,5,6,7,8,9])
all_x = x[:]
remove_first = x[1:]
btn = x[1:3]

# skip 2
skip_2 = x[::2]

# skip in reverse order
skip_reverse = x[::-1]

# getting specific indices list
indices_list = tf.constant([1,3])
x = tf.gather(x, indices_list)

"""
reshaping
"""
x = tf.range(9)

x = tf.reshape(x, (3,3))

# transpose
x = tf.transpose(x)
print(x)








