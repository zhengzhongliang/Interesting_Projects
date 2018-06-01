import tensorflow as tf
import numpy as np

#t1 = [[1, 2, 3], [4, 5, 6]]
#t2 = [[7, 8, 9], [10, 11, 12]]
#C = tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

A = tf.constant(np.zeros((16,16,16)))

B = tf.concat([B,A],axis=0)

with tf.Session() as sess:
  print(sess.run(B))
