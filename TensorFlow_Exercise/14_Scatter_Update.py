import numpy as np
import tensorflow as tf

A = tf.get_variable(name='A', shape=[16,16], dtype=None, initializer=None)

B = tf.scatter_update(A,[0],np.zeros(16).reshape((1,16)))


print(A)
init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  print(sess.run(B))
