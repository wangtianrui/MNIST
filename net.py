import tensorflow as tf
import function


def net(input, batch_size, num_class, keep_prob, name):
    with tf.variable_scope(name) as scope:
        conv1 = function.conv(input=input, ksize=[5, 5, 3, 32], strides=[1, 1, 1, 1], name='conv1')
        pool1 = function.maxPool(input=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool1')
        print(pool1)
        conv2 = function.conv(input=pool1, ksize=[5, 5, 32, 64], strides=[1, 1, 1, 1], name='conv2')
        pool2 = function.maxPool(input=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool2')
        print(pool2)
        local1 = function.localFCWithDropout(pool2, 1024, batch_size=batch_size,keep_prob=keep_prob, name="local1")
        print(local1)
        softmax = function.sotfMax(local1, n_class=num_class)
        print(softmax)

        return softmax
