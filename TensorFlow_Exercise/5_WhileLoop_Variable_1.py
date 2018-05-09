import numpy as np
import tensorflow as tf

import collections


class RNN:


  def __init__(self):
    def body_(i,x,y):
      return (tf.add(i,tf.constant(1)), tf.add(x,y), tf.subtract(x,y))
  
    init = tf.constant(np.random.rand(3))
    self.x = tf.get_variable(name='x', initializer=init)
    self.y = tf.get_variable(name='y', initializer=init)
    
    #self.i = tf.get_variable(name='i', initializer=tf.constant(1))
    self.i = tf.constant(1)
   
    cond = lambda i, x, y: tf.less(i,10)
    body = lambda i, x, y: body_(i,x,y)

    (self.i, self.x, self.y)=tf.while_loop(cond, body, (self.i, self.x, self.y))

  


rnn = RNN()


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(sess.run([rnn.i, rnn.x, rnn.y]))
    
