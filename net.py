import tensorflow as tf
import function


def net(input, batch_size, name):
    with tf.variable_scope(name) as scope:
        conv1 = function.conv(input=input, ksize=[5, 5, 3, 32], name='conv1')
        pool1 = function.maxPool(input=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool1')

        conv2 = function.conv(input=pool1, ksize=[5, 5, 32, 64], name='conv2')
        pool2 = function.maxPool(input=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool2')

        local1 = function.localFC(pool2, 1024, batch_size=batch_size, name="local1")
        softmax = function.sotfMax(local1, batch_size=batch_size)

        return softmax
