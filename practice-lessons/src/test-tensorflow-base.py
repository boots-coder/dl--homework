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
# c = tf.eye(3)
# tf.shape(c)
# # return numbers of c
# tf.size(c)
# # t = [
# #     [
# #         [1, 1, 1],
# #         [2, 2, 2]
# #     ],
# #      [
# #         [3, 3, 3],
# #         [4, 4, 4]
# #      ]
# #      ]
# # tf.shape(t)
# # a = [[[0, 0, 1], [0 ,1 , 0],[1, 0, 0]]]
# # tf.rank(a)
#
# t=[1, 2, 3, 4, 5, 6, 7, 8, 9]
# tf.reshape(t, [3,3])
# # the same as 3, -1
# tf.reshape(t, [3,-1])
#
# t = [1, 2, 2, 3]
# tf.shape(t)
# # t = tf.expand_dims(t, 0)
# print("tf.shape(t):", tf.shape(t))
#
#
# # about slide fun
#
# input = tf.constant([[1, 2, 3],
#                      [4, 5, 6],
#                      [7, 8, 9]])
#
# begin = [0, 0]  # 从 (0,0) 开始
# size = [2, 2]   # 提取 2x2 的子矩阵
#
# result = tf.slice(input, begin, size)
#
# a = tf.random.normal([5, 30], seed=43)
# a = tf.random.uniform([5, 30],minval = 1, maxval = 10, seed=43,dtype = tf.int32)
#
# print(a)
# s0, s1, s2 = tf.split(a, 3 , 1)
# s0
# transpose of the matrix

# 设置全局随机种子
# tf.random.set_seed(43)
# a = tf.random.uniform([2, 4], minval = 1, maxval = 10 , dtype=tf.int32)
# print(a)
# transpose_a = tf.transpose(a)
# print(transpose_a)
# # onehot coding
# indices = [0, 1, -1 ,2]
# tf.one_hot(indices,depth = 5, on_value=1, off_value=0)

# # 矩阵运算
# diagonal = [1. ,2. ,3. ,4.]
# matrix = tf.linalg.diag(diagonal)
# tf.linalg.det(matrix)
# tf.linalg.trace(matrix)
# tf.linalg.inv(matrix)
# tf.linalg.det(tf.linalg.inv(matrix))*tf.linalg.det(matrix)
#
# a = tf.constant([1.,2.,3.,4., 5, 6.])
# a = tf.reshape(a, [2, 3])
# b = tf.reshape(a, [3, 2])
# tf.linalg.matmul(a, b)
# if tf.reduce_all(tf.equal(a, tf.transpose(b))):
#     print("矩阵 a 等于矩阵 b 的转置。")
#     print("a:")
#     print(a)
#     print("\nb 的转置:")
#     print(tf.transpose(b))
# else:
#     print("矩阵 a 不等于矩阵 b 的转置。")
#     print("a:")
#     print(a)
#     print("\nb 的转置:")
#     print(tf.transpose(b))

# 复数操作
# tensor ‘real’ is [2.25, 3.25]
real = [2.25, 3.25]
# tensor imag is [4.75, 5.75]
imag = [4.75, 5.75]
tf.complex(real, imag)





