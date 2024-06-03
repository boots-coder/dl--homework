import numpy as np
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt

# 载入示例图像
china = load_sample_image("china.jpg")

# 选取图像的部分区域
image = china[150:220, 130:250]

# 获取图像的高度、宽度和颜色通道数
height, width, channels = image.shape

# 将彩色图像转换为灰度图像
image_grayscale = image.mean(axis=2).astype(np.float32)

# 更改图像维度，为了使用在某些特定的机器学习或深度学习框架中
images = image_grayscale.reshape(1, height, width, 1)
"""
在使用 matplotlib 的 imshow() 函数展示图像时，cmap 参数指的是“colormap”，即颜色映射表。对于灰度图像，我们通常使用 cmap='gray' 参数来确保图像以灰度（黑白）方式显示，而不是使用默认的颜色映射。

如果不指定 cmap='gray'，则 matplotlib 会使用默认的颜色映射，这通常是蓝紫到黄色的渐变，这样的显示对于灰度图像来说通常不是我们期望的视觉效果。因此，当你处理灰度图像并希望以真实的灰度级显示时，使用 cmap='gray' 是很重要的。

如果你希望看到灰度图像在不同颜色映射下的效果，可以尝试改变 cmap 参数，例如使用 cmap='viridis' 或 cmap='magma' 等，这些都是 matplotlib 提供的其他颜色映射选项。
"""
# 可以使用 matplotlib 查看处理后的灰度图像
plt.imshow(image_grayscale ,cmap='gray')
plt.show()