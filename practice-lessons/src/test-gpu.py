import tensorflow as tf
import time

# 定义一个简单的矩阵乘法函数
def mul(x, steps):
    for i in range(steps):
        x = tf.matmul(x, x)

# 测试 CPU 计算时间
with tf.device("/cpu:0"):
    start_time = time.time()
    mul(tf.random.normal((1000, 1000)), 1000)
    cpu_duration = time.time() - start_time
    print("CPU Duration: {:.2f} seconds".format(cpu_duration))

# 测试 GPU 计算时间
if tf.config.list_physical_devices("GPU"):
    with tf.device("/gpu:0"):
        start_time = time.time()
        mul(tf.random.normal((1000, 1000)), 1000)
        gpu_duration = time.time() - start_time
        print("GPU Duration: {:.2f} seconds".format(gpu_duration))
else:
    print("GPU: not found")