import numpy as np
import tensorflow as tf

#def true_func(x,y):
#  return tf.add(x,y)

#def false_func(x,y):
#  return tf.subtract(x,y)


#x = tf.constant(5)
#y = tf.constant(6)

#a = tf.cond(tf.less(x,y), lambda:true_func(x,y), lambda:false_func(x,y))


def cond(x,y):
  return tf.less(x,y)

def body(x,y):
  return tf.add(x,tf.constant(1)), y

x = tf.constant(2)
y = tf.constant(6)

final = tf.while_loop(cond,body, [x,y])

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(sess.run(final))
