import numpy as np
import tensorflow as tf
"""
california hosing data prediction with linear model
"""
# from sklearn.datasets import fetch_california_housing
# housing = fetch_california_housing()
# m, n = housing.data.shape
# #np.c_按colunm来组合array
# housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
#
# X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
# XT = tf.transpose(X)
# theta = tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(XT, X)), XT), y)
#
# print(theta)

"""
minist data detection 
"""

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 归一化
x_train, x_test = x_train / 255.0, x_test / 255.0

# build model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),  # 将28x28的输入图像展平成一维向量
  tf.keras.layers.Dense(64, activation='relu'),  # 全连接层，128个神经元，ReLU激活函数
  tf.keras.layers.Dropout(0.2),                   # Dropout层，随机丢弃20%的神经元，防止过拟合
  tf.keras.layers.Dense(10, activation='softmax') # 全连接层，10个神经元，Softmax激活函数，用于分类
])
# model compile
model.compile(
    optimizer='adam',  # 使用 Adam 优化器
    loss='sparse_categorical_crossentropy',  # 使用稀疏分类交叉熵损失函数
    metrics=['accuracy']  # 使用准确率作为评估指标
)

# train

model.fit(x_train, y_train, epochs=5)

# verify

model.evaluate(x_test,  y_test, verbose=2)




