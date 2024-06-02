import tensorflow as tf
input = tf.random.normal([1,13,5,5])
filter = tf.random.normal([6,3,5,2])
op1 = tf.nn.conv2d( input, filter,
strides=[1, 5, 2, 1], padding='SAME')
print(op1.shape)

op2 = tf.nn.conv2d(input, filter, strides=[1, 5, 2, 1], padding='VALID')

print(op2.shape)


