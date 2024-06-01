import tensorflow as tf

opt = tf.keras.optimizers.SGD(learning_rate=0.1)
var = tf.Variable(1.0)
loss = lambda: (var ** 2)/2.0

'''
在 TensorFlow 2.x 中，tf.keras.optimizers.SGD优化器没有 minimize 方法。
相反，你可以使用 apply_gradients 方法来应用计算的梯度。
你需要手动计算梯度，然后将其应用到变量。
'''
# 使用 GradientTape 计算梯度并应用梯度
with tf.GradientTape() as tape:
    loss_value = loss()
grads = tape.gradient(loss_value, [var])
opt.apply_gradients(zip(grads, [var]))

# 输出变量的值
print(var.numpy())

y_true = tf.constant([[0., 1.], [0., 0.]])
y_pred = tf.constant([[1., 1.], [1., 0.]])
mse = tf.keras.losses.MeanSquaredError()
mse(y_true, y_pred).numpy()


y_true = tf.constant([0, 1, 0, 0])
y_pred = tf.constant([-18.6, 0.51, 2.94, -12.8])
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
bce(y_true, y_pred).numpy()

m = tf.keras.metrics.Accuracy()
m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]])
m.result().numpy()

m.reset_state()
m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]], sample_weight=[1, 1, 0, 0])
m.result().numpy()

# model.compile(optimizer='sgd',
#               loss='mse',
#               metrics=[tf.keras.metrics.Accuracy()])

m.reset_state()
m.update_state([[1], [1], [0], [0]], [[0.98], [1], [0], [0.6]],
               sample_weight=[1, 0, 0, 1])
m.result().numpy()
