import tensorflow as tf
import numpy as np

x = [[[1, 2],
      [3, 4]],
     [[5, 6],
      [7, 8]]]


y = tf.reduce_mean(x,2)


sess = tf.Session()

print(sess.run(y))