import tensorflow as tf

# # 转换到v1 并创建计算图
# tf.compat.v1.disable_v2_behavior()
# a = tf.constant([1, 2])
# b = tf.constant([3, 4])
# c = tf.add(a, b)
# d = a * b
# # 执行计算图
# sess = tf.compat.v1.Session()
# print(sess.run(c))
# print(sess.run(d))
# sess.close()

# tf.compat.v1.enable_v2_behavior()
# 使用tf 2

a = tf.constant(1.)
b = tf.constant(2.)
print("a+b=", a + b)

e = 2
str = "hello world value is {value} "
print(str.format(value = e))
input = [[[1, 1 ], [ 2, 2]],[[ 3, 3] ,[4, 4]],[[5, 5], [ 6, 6]]]
t = tf.slice(input, [1, 0, 0], [1, 1, 1])
print(t)

import tensorflow as tf

# 原始张量
tensor = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 在第 0 维增加一个维度
expanded_tensor = tf.expand_dims(tensor, axis=2)

print(expanded_tensor)
print(expanded_tensor.shape)