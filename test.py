import tensorflow as tf
import numpy as np


x = 1


y = 1

correct = tf.equal(x,y)
correct = tf.cast(correct, tf.float32)
sess = tf.Session()


print(sess.run(correct))