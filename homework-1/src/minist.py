import tensorflow as tf
import numpy as np
import datetime
from tensorflow.keras.callbacks import TensorBoard

# 加载 MNIST 数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 归一化
x_train, x_test = x_train / 255.0, x_test / 255.0

# 创建 trainX1k 数据集
trainX1k = []
trainY1k = []
for digit in range(10):
    idx = np.where(y_train == digit)[0][:100]
    trainX1k.extend(x_train[idx])
    trainY1k.extend(y_train[idx])

trainX1k = np.array(trainX1k)
trainY1k = np.array(trainY1k)

print("trainX1k shape:", trainX1k.shape)
print("trainY1k shape:", trainY1k.shape)

# 定义模型构建和编译函数
def build_and_compile_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 训练并记录到 TensorBoard 的函数
def train_and_log(model, x_train, y_train, x_test, y_test, log_dir):
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(
        x_train, y_train, epochs=50,
        validation_data=(x_test, y_test),
        callbacks=[tensorboard_callback]
    )

# 训练使用 trainX1k 的模型
model_1k = build_and_compile_model()
log_dir_1k = "logs/fit/trainX1k_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_and_log(model_1k, trainX1k, trainY1k, x_test, y_test, log_dir_1k)

# 训练使用 x_train 的模型
model_full = build_and_compile_model()
log_dir_full = "logs/fit/trainX_full_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_and_log(model_full, x_train, y_train, x_test, y_test, log_dir_full)