import tensorflow as tf

a = tf.add(tf.constant(1),tf.constant(1))

with tf.Session() as sess:
  print(sess.run(a))
