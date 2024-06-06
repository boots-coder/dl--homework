
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical

# 加载数据集并进行预处理
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 因为 MNIST 是黑白图片，所以我们需要增加一个维度来表示单通道
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 归一化数据
train_images, test_images = train_images / 255.0, test_images / 255.0

# 将标签转换为 one-hot 编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# 导入tensorflow
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 构建神经网络
model = models.Sequential([
    tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding="SAME", input_shape=(28, 28, 1), name="C1"),  # 卷积层1(对于图中C1)，卷积核5x5
    tf.keras.layers.MaxPooling2D((2, 2), strides=2, name="S2"),  # 池化层1(S2)，2x2采样
    tf.keras.layers.Dropout(rate=0.25, name='drop1'),  # dropout

    tf.keras.layers.Conv2D(32, (2, 2), activation='relu', padding="SAME", name="C3"),  # 卷积层2(C3)，卷积核5x5
    tf.keras.layers.MaxPooling2D((2, 2), strides=2, name="S4"),  # 池化层2(S4)，2x2采样
    tf.keras.layers.Dropout(rate=0.25, name='drop2'),  # dropout

    tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding="SAME", name="C5"),  # 卷积层2(C5)，卷积核5x5
    tf.keras.layers.MaxPooling2D((2, 2), strides=2, name="S6"),  # 池化层2(S6)，2x2采样
    tf.keras.layers.Dropout(rate=0.25, name='drop3'),  # dropout

    # tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding="SAME", name="C7"),  # 卷积层2(C7)，卷积核5x5
    # tf.keras.layers.MaxPooling2D((2, 2), strides=2, name="S8"),  # 池化层2(S8)，2x2采样
    # tf.keras.layers.Dropout(rate=0.25, name='drop4'),  # dropout

    tf.keras.layers.Flatten(),  # Flatten层，连接卷积层与全连接层
    tf.keras.layers.Dense(128, activation='relu', name="F9"),  # 全连接层，特征进一步提取
    tf.keras.layers.Dense(10, activation='softmax', name="Out")  # 输出层，输出预期结果
])

# 打印网络结构
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = tf.reshape(x_train, (60000, 28, 28, 1))
x_test = tf.reshape(x_test, (10000, 28, 28, 1))

model.fit(
    x_train, y_train, epochs=5,
    verbose=2
)

scores = model.evaluate(x_test, y_test, verbose=2)
print("Accuracy: %.2f%%" % (scores[1] * 100))
print("Loss: %.2f" % scores[0])
