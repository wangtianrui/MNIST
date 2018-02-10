import tensorflow as tf

img_w = 28
img_h = 28
n_class = 10


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


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


def conv2(input, ksize, strides, name, padding='SAME'):
    with tf.variable_scope(name) as scope:
        weights = weight_variable(ksize)

        biases = bias_variable(ksize[3])

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
        print("test:", dim)
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

def localFCWithDropout2(input, outChannels, batch_size, keep_prob, name):
    with tf.variable_scope(name) as scope:
        # reshape(t, [-1]) == > [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
        reshape = tf.reshape(input, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = weight_variable([dim,outChannels])
        biases = bias_variable([outChannels])
        local = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        drop = tf.nn.dropout(local, keep_prob=keep_prob, name="dropout")
        return drop


def sotfMax(input, n_class, name='softmax'):
    dim = input.get_shape()[1].value
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights',
                                  shape=[dim, n_class],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_class],
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
    print(image)
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
            capacity=2000,
            min_after_dequeue=1500
        )
    else:
        image_batch, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=64,
            capacity=2000
        )

    # 生成一个[batch_size,n_class]的稀疏矩阵，用来表示label
    """
    [ 1,0,0,0,0,0,0,0,0,0 ] 表示label为0
    [ 0,1,0,0,0,0,0,0,0,0 ] 表示label为1
    """
    label_batch = tf.one_hot(label_batch, depth=n_class)
    label_batch = tf.cast(label_batch, dtype=tf.float32)
    label_batch = tf.reshape(label_batch, [batch_size, n_class])
    #label_batch = tf.cast(label_batch,tf.float32)
    return image_batch, label_batch


def loss(logits, labels):
    with tf.name_scope('loss') as scope:
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy')
        cross_entropy =  tf.reduce_mean(-tf.reduce_sum(labels*tf.log(logits), reduction_indices=[1]))
        #cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
        #loss = tf.reduce_mean(cross_entropy, name='loss')
        #tf.summary.scalar(scope + "/loss", loss)
        return cross_entropy


def optimize(loss, learning_rate, global_step):
    with tf.name_scope("optimize"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=global_step)
        return train_op


def accuracy(logits, labels):
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct) * 100.0
        tf.summary.scalar(scope + "/accuracy", accuracy)
        return accuracy
