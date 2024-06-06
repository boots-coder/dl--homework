import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
def generate_time_series(batch_size, n_steps):
    np.random.seed(43)
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., np.newaxis].astype(np.float32)

n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train, y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
X_test, y_test = series[9000:, :n_steps], series[9000:, -10:, 0]

# print(X_train.shape)
# 绘制时间序列
plt.figure(figsize=(10, 4))
#选取x 的第0 个样本的所有列的值进行画图
plt.plot(X_train[0, :, 0], marker='o', label='Sample Time Series')
plt.title('Time Series Sample')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.show()



# y_valid = tf.constant(y_valid)
#
# y_pred = tf.constant(X_valid[:, -1])
#
# mse = keras.losses.MeanSquaredError()
#
#  tf.Tensor(0.020891849, shape=(), dtype=float32)
# print(mse(y_valid, y_pred))
#
# model = keras.models.Sequential([
#   keras.layers.SimpleRNN(1, input_shape=[None, 1])
# ])
#
# model.compile(loss='mse', optimizer='adam')
#
# model.fit(X_train, y_train,)

Y = np.empty((10000, n_steps, 10)) # each target is a sequence of 10D vectors
print(Y.shape)
for step_ahead in range(1, 10 + 1):
    Y[:, :, step_ahead - 1] = series[:, step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]
print(Y_train.shape)
model = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20, return_sequences=True),

    keras.layers.TimeDistributed(keras.layers.Dense(10))
])

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, Y_train, epochs=1, validation_data=(X_valid, Y_valid))

scores = model.evaluate(X_test, Y_test)

print('mas for test', scores)
