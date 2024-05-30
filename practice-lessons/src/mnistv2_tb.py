import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 定义模型
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 定义 TensorBoard 回调
log_dir = "logs/fit/mnist"
callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 训练模型
model.fit(x_train, y_train, epochs=10, callbacks=[callback])  # 改为 5 个 epoch 以加快测试
model.evaluate(x_test, y_test, verbose=2)
