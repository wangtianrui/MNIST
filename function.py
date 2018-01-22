import tensorflow as tf

img_w = 28
img_h = 28
n_class = 10


def conv(input, ksize, strides, name, padding='SAME'):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights',
                                  shape=ksize,
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[ksize[3]],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(input, filter=weights, strides=strides, padding=padding)
        relu = tf.nn.relu(tf.nn.bias_add(conv, bias=biases), name=scope.name)

        return relu


def maxPool(input, ksize, strides, name, padding='SAME'):
    with tf.variable_scope(name) as scope:
        pool = tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding, name=name)
        return pool


def maxPoolAndNormal(input, ksize, strides, name, padding='SAME'):
    with tf.variable_scope(name) as scope:
        pool = tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding, name=name)
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
        return drop


def sotfMax(input, batch_size, name='softmax'):
    dim = input.get_shape()[1].value
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights',
                                  shape=[dim, batch_size],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[batch_size],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        # softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), dim)

        return tf.nn.softmax(tf.matmul(input, weights) + biases)


def readDataFromTF(filename, batch_size, shuffle=True):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized=serialized_examples,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        }
    )
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [img_w, img_h, 3])
    # 将数据类型转换
    image = tf.cast(image, tf.float32)
    label = tf.cast(img_features['label'], tf.int32)
    if shuffle:
        # min_after_dequeue，一定要保证这参数小于capacity参数的值，否则会出错。这个代表队列中的元素大于它的时候就输出乱的顺序的batch。
        # capacity可以看成是局部数据的范围，读取的数据是基于这个范围的，在这个范围内，min_after_dequeue越大，数据越乱
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=64,
            capacity=20000,
            min_after_dequeue=15000
        )
    else:
        image_batch, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=64,
            capacity=2000
        )
