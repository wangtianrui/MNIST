import os
import os.path
import math
import numpy as np
import tensorflow as tf
import cifar10_input
import function
import net


BATCH_SIZE = 32
def evaluate():
    with tf.Graph().as_default():

        log_dir = 'F:/Traindata/MNIST_pictures/result'
        test_dir = 'E:/python_programes/My-TensorFlow-tutorials/02 CIFAR10/data/'
        n_test = 1000

        # reading test data
        images, labels = cifar10_input.read_cifar10(data_dir=test_dir,
                                                    is_train=False,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False)

        logits = net.inference(images)
        print("labels", labels)
        print("logits", logits)
        accuracy221 = function.accuracy22(logits, labels)
        print(accuracy221)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            accuracy2212 = sess.run([accuracy221])
            accuracy2212 = accuracy2212[0]
            try:
                num_iter = int(math.ceil(n_test / BATCH_SIZE))
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0

                while step < num_iter and not coord.should_stop():

                    #true_count += np.sum(predictions)
                    step += 1
                    #precision = true_count / total_sample_count
                print(accuracy2212)
                print('accuracy22: %.4f%% *****'
                      % (accuracy2212))
                #print('precision = %.3f' % )
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)


# %%
evaluate()