import tensorflow as tf
import numpy as np
import os
import net
import function

n_class = 10
batch_size = 32
img_w = 28
img_h = 28
LEARNING_RATE = 0.001
MAX_STEP = 5000

CAPACITY = 2000

TRAIN_FILENAME = "F:/Traindata/MNIST_pictures/MNISTTrain.tfrecords"
TEST_FILENAME = "F:/Traindata/MNIST_pictures/MNISTTest.tfrecords"

TRAIN_RESULTS_FILENAME = "F:/Traindata/MNIST_pictures/result"


# 1
def train():
    train_image_batch, train_label_batch = function.readDataFromTF(TRAIN_FILENAME, batch_size=batch_size)

    logits = net.net(train_image_batch, batch_size=batch_size, num_class=n_class, keep_prob=0.5, name="train")
    #train_logits = net.net(train_image_batch, batch_size=batch_size, num_class=n_class, keep_prob=1.0,name="train_accuracy")
    train_logits = net.inference(train_image_batch)

    print("int?", train_label_batch)
    loss = function.losses(logits=logits, labels=train_label_batch)
    # loss = tf.reduce_mean(-tf.reduce_sum(train_label_batch * tf.log(logits), reduction_indices=[1]))
    #train_accuracy = function.accuracy(train_logits, train_label_batch)

    my_global_step = tf.Variable(0, name='global_step')
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train_op = optimizer.minimize(loss, global_step=my_global_step)
    # train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    tra_summary_writer = tf.summary.FileWriter(TRAIN_RESULTS_FILENAME, sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break

            _, train_loss = sess.run([train_op, loss])

            if (step % 50 == 0) or (step == MAX_STEP):
                print('***** Step: %d, loss: %.4f' % (step, train_loss))
                summary_str = sess.run(summary_op)
                tra_summary_writer.add_summary(summary_str, step)
            if (step % 200 == 0) or (step == MAX_STEP):
                #trainaccuracy = sess.run([train_accuracy])
                print('***** Step: %d, loss: %.4f*****'
                      % (step, train_loss))
                summary_str = sess.run(summary_op)
                tra_summary_writer.add_summary(summary_str, step)
            if step % 2000 == 0 or step == MAX_STEP:
                checkpoint_path = os.path.join(TRAIN_RESULTS_FILENAME, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('error')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


train()
