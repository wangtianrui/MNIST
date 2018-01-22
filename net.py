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
        local1 = function.localFCWithDropout(pool2, 1024, batch_size=batch_size, keep_prob=keep_prob, name="local1")
        print(local1)
        with tf.variable_scope('softmax_linear') as scope:
            weights = tf.get_variable('softmax_linear',
                                      shape=[1024, num_class],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[num_class],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            softmax = tf.add(tf.matmul(local1, weights), biases, name='softmax_linear')
        print(softmax)

        return softmax
