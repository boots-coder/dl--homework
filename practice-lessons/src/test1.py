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
