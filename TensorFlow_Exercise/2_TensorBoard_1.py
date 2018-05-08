import numpy as np
import tensorflow as tf

g1 = tf.Graph()
with g1.as_default() as g:
  with tf.name_scope(name='g1') as scope:
    m1 = tf.constant([[3., 2.]], name='m1')
    m2 = tf.constant([[3.],[3.]], name='m2')
    m3 = tf.matmul(m1, m2, name='m3')
tf.reset_default_graph()

writer=tf.summary.FileWriter('event')
writer.add_graph(g1)


with tf.Session(graph=g1) as sess:
  sess.run([m1,m2,m3])



# command to be typed in terminal: tensorboard --logdir="event"

# notes:
# 1: tensorboard log will be saved only after session is run
# 2: logdir should not be a file but a path
