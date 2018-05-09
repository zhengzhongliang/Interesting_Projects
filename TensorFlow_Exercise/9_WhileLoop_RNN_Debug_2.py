import numpy as np
import tensorflow as tf

class RNN():
    
    


  def __init__(self, time_step, input_size, hidden_size, output_size):
    self.x = tf.placeholder(shape=[time_step, input_size] ,dtype=tf.float32)
    self.y_ = tf.placeholder(shape=[time_step, output_size] ,dtype=tf.float32)
    self.h = tf.get_variable(name='h', shape=[time_step, hidden_size],dtype=tf.float32)
    self.y = tf.get_variable(name='y', shape=[time_step, output_size],dtype=tf.float32)

    self.U = tf.get_variable(name='U',shape=[input_size, hidden_size],dtype=tf.float32)   # Change Initilizer Here!
    self.V = tf.get_variable(name='V',shape=[hidden_size, output_size],dtype=tf.float32)
    self.W = tf.get_variable(name='W',shape=[hidden_size, hidden_size],dtype=tf.float32)

    self.i = tf.constant(0) 

    def loop_body_def(i, x,y,h, U , V, W):

      h=tf.assign(h[i,:],tf.constant(np.ones(3))) 
      return tf.add(i,tf.constant(1)), x,y,h, U , V, W
      

    loop_cond = lambda i, x, y, h,U,V,W:tf.less(i, tf.constant(time_step-1))
    loop_body = lambda i, x, y, h,U,V,W:loop_body_def(i, x,y,h,U,V,W)

    self.i,self.x, self.y, self.h,self.U, self.V, self.W=tf.while_loop(loop_cond, loop_body, [self.i,self.x, self.y, self.h,self.U, self.V, self.W])


rnn = RNN(time_step=5, input_size=4, hidden_size=3, output_size=2)

X = np.ones((5,4))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(sess.run([rnn.h,rnn.y,rnn.i], feed_dict={rnn.x:X}))
