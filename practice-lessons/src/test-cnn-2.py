import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

china = load_sample_image('china.jpg')/255
flower = load_sample_image('flower.jpg')/255
# print(china.shape) (427, 640, 3)
print(china)
china = china[150:220, 130:250]
flower = flower[150:220, 130:250]

# china = china.mean(axis=2).astype(np.float32)

images = np.array([china, flower])
batch_size, height, width, channels = images.shape
filters = np.zeros(
    shape=(7, 7, channels , 2),
    dtype=np.float32
)
filters[:, 3, :, 0] = 1 # 横的
filters[3, :, :, 1] = 1 # 竖的

plt.imshow(filters[:, :, :, 0], cmap='gray')
plt.show()
plt.imshow(filters[:, :, :, 1], cmap='gray')
plt.show()

# build outputs
outputs = tf.nn.conv2d(
    images,
    filters,
    strides=1,
    padding='SAME'
)
# keras.layers.Conv2D(
#      filters,        # 滤波器的数量
#      kernel_size,    # kernel的高度和宽度
#                      # 1个或2个整数（tuple、list）
#      strides=(1,1),
#      padding='valid',
#      data_format=None,
#      …
# )

plt.imshow(outputs[0, :, :, 0], cmap='gray')
plt.show()

plt.imshow(outputs[0, :, :, 1], cmap='gray')
plt.show()