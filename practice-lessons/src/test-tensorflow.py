import time

import tensorflow as tf


# def mul(x, steps):
#     start_time = time.time()
#     for i in range(steps):
#         x = tf.matmul(x, x)
#     end_time = time.time() - start_time
#     print("finished in {} seconds".format(end_time))
#     return x


# # 指定在CPU中运行:
# with tf.device("/cpu:0"):
#     mul(tf.random.normal((1000, 1000)), 1000)
# # 指定在GPU中运行:
# if tf.config.list_physical_devices("GPU"):
#     with tf.device("/gpu:0"):
#         mul(tf.random.normal((1000, 1000)), 1000)
# else:
#     print("GPU: not found")

# var = tf.__version__
# print(var)
# # create 3*4
# tf.zeros([3, 4], dtype=tf.int32)
# tf.ones([3, 4], dtype=tf.int32)
# tf.fill([3, 3], 9)
# tf.eye(3)
#
# # create a constant
# tf.constant(1.)
# # create a variable with shape [3, 4]
# tf.Variable(tf.ones([3, 4]))
#
# # random  numbers and set a seed
# tf.random.set_seed(42)
# tf.random.normal([3, 4])
#
# # you can also combine like this
# tf.random.normal([3, 4], seed=42)
#
# # some function for math
# a = tf.constant([1., 2., 3.])
# print(tf.exp(a))
#
# # contact , the using of axis
# a = tf.constant([[1., 2.], [3., 4.]])
# b = tf.constant([[5., 6.], [7., 8.]])
#
# print(tf.concat([a, b], axis=0))
# print(tf.concat([a, b], axis=1))

# data structor cast
# a = tf.constant([1.4, 2.8, 3.0])
# b = tf.cast(tf.constant([1.4, 2.8, 3.0]), dtype=tf.int32)
# print(b)
# test of shape and size
# tf.shape(b)
c = tf.eye(3)
tf.shape(c)
# return numbers of c
tf.size(c)
t = [
    [
        [1, 1, 1],
        [2, 2, 2]
    ],
     [
        [3, 3, 3],
        [4, 4, 4]
     ]
     ]
# tf.shape(t)
# a = [[[0, 0, 1], [0 ,1 , 0],[1, 0, 0]]]
# tf.rank(a)

t=[1, 2, 3, 4, 5, 6, 7, 8, 9]
tf.reshape(t, [3,3])
# the same as 3, -1
tf.reshape(t, [3,-1])



