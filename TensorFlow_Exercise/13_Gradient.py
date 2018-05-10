import tensorflow as tf
import numpy as np

A = tf.constant(np.ones((1,5)), dtype=tf.float32)
W = tf.get_variable(name='W', shape=[5,3], dtype=tf.float32)
B = tf.constant(np.ones((1,3))*0.5, dtype=tf.float32)

output = tf.matmul(A,W)
loss = tf.losses.mean_squared_error(labels=B, predictions=output)

g_w = tf.gradients(loss,[W])

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  g_w_eval = sess.run(g_w)

  for g in g_w_eval:
    print(g)
