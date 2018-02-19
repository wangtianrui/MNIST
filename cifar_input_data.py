from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow.python.platform import gfile

# 图片的长宽
IMAGE_SIZE = 32
# 图片的种类
NUM_CLASSES = 10

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def input_from_binary(data_dir, batch_size):
    if not data_dir:
        raise ValueError('找不到data的文件目录')
    return distorted_inputs(data_dir=data_dir,
                            batch_size=batch_size)


# 读取二进制图片
def distorted_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in range(1, 6)]
    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # 创建一个队列来读取
    filename_queue = tf.train.string_input_producer(filenames)

    # 从队列中读取数据
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    # height = IMAGE_SIZE
    # width = IMAGE_SIZE


    # distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # 随机水平移动
    # distorted_image = tf.image.random_flip_left_right(distorted_image)

    # 随机排序
    # distorted_image = tf.image.random_brightness(reshaped_image,max_delta=63)
    # distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)

    # 减去均值像素，并除以像素方差 (图片标准化)
    float_image = tf.image.per_image_standardization(reshaped_image)

    # 确保随机洗牌具有良好的混合性能.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # 通过建立一个队列实例生成一批图像和标签
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size)


def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    # 图片原大小
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # 总大小
    record_bytes = label_bytes + image_bytes

    # 从二进制文件中读取固定长度记录
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    # decode_raw操作可以将一个字符串转换为一个uint8的张量。
    record_bytes = tf.decode_raw(value, tf.uint8)

    # 首个字节是label,并将uint8转换成int32
    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    # 剩下的是图片, 从[depth*height*width]变成 [depth, height, width]
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    # [depth, height, width] -> [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
    """构建batch
    输入:
      image: [height, width, 3]
      label: [label]

    Returns:
      images: [1, height, width, 3]
      labels: [1]
    """
    # 创建一个队列打乱数据, 然后读取从队列中'batch_size' images + labels .
    num_preprocess_threads = 16

    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    tf.summary.image('images', images)

    labels = tf.reshape(label_batch, [batch_size])

    sparse_labels = tf.reshape(labels, [batch_size, 1])

    indices = tf.expand_dims(tf.range(0, batch_size), 1)

    # 组合

    concated = tf.concat([indices, sparse_labels], 1)

    dense_labels = tf.sparse_to_dense(concated,
                                      [batch_size, NUM_CLASSES],
                                      1.0, 0.0)

    return images, dense_labels
