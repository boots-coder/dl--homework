import time

import tensorflow as tf
def mul(x, steps):
    start_time = time.time()
    for i in range(steps):
        x = tf.matmul(x, x)
    end_time = time.time() - start_time
    print("finished in {} seconds".format(end_time))
    return x
# 指定在CPU中运行:
with tf.device("/cpu:0"):
    mul(tf.random.normal((1000,1000)), 1000)
# 指定在GPU中运行:
if tf.config.list_physical_devices("GPU"):
    with tf.device("/gpu:0"):
        mul(tf.random.normal((1000,1000)), 1000)
else:
    print("GPU: not found")

