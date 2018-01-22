import tensorflow as tf


def conv(input, shape, strides, name, padding='SAME'):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights',
                                  shape=shape,
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[shape[3]],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(input, filter=weights, strides=strides, padding=padding)
        relu = tf.nn.relu(tf.nn.bias_add(conv, bias=biases), name=scope.name)

        return relu


def maxPool(input, shape, strides, name, padding='SAME'):
    with tf.variable_scope(name) as scope:
        pool = tf.nn.max_pool(input, ksize=shape, strides=strides, padding=padding, name=name)
        return pool


def maxPoolAndNormal(input, shape, strides, name, padding='SAME'):
    with tf.variable_scope(name) as scope:
        pool = tf.nn.max_pool(input, ksize=shape, strides=strides, padding=padding, name=name)
        normal = tf.nn.lrn(pool, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='normal')
        return normal


def localFC(input, outChannels, batch_size, name):
    with tf.variable_scope(name) as scope:
        # reshape(t, [-1]) == > [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
        reshape = tf.reshape(input, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, outChannels],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))

        biases = tf.get_variable('biases',
                                 shape=[outChannels],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        local = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        return local


def localFCWithDropout(input, outChannels, batch_size, keep_prob, name):
    with tf.variable_scope(name) as scope:
        # reshape(t, [-1]) == > [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
        reshape = tf.reshape(input, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, outChannels],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))

        biases = tf.get_variable('biases',
                                 shape=[outChannels],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        local = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        drop = tf.nn.dropout(local, keep_prob=keep_prob, name="dropout")
        return dropl
