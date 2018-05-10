import numpy as np
import tensorflow as tf

class RNN():
    
    


  def __init__(self, time_step, input_size, hidden_size, output_size):
    self.x = tf.placeholder(shape=[time_step, input_size] ,dtype=tf.float32)
    self.y_ = tf.placeholder(shape=[time_step, output_size] ,dtype=tf.float32)
    self.h = list()
    self.y = list()

    self.U = tf.get_variable(name='U',shape=[input_size, hidden_size],dtype=tf.float32)   # Change Initilizer Here!
    self.V = tf.get_variable(name='V',shape=[hidden_size, output_size],dtype=tf.float32)
    self.W = tf.get_variable(name='W',shape=[hidden_size, hidden_size],dtype=tf.float32)

    self.i = tf.constant(0) 

    for i in np.arange(time_step):
      if(i==0):
        h_current = tf.sigmoid(tf.matmul(tf.reshape(self.x[i,:],shape=[1, input_size]),self.U))
        y_current = tf.matmul(h_current, self.V)    # y_current(1,2)
        self.h.append(h_current)
        self.y.append(y_current)
      else:
        h_prev = tf.reshape(self.h[i-1], shape=[1, hidden_size])
        x_current= tf.reshape(self.x[i,:],shape=[1, input_size])
        h_current = tf.sigmoid(tf.add(tf.matmul(x_current,self.U), tf.matmul(h_prev, self.W)))
        y_current = tf.matmul(h_current, self.V)
        self.h.append(h_current)
        self.y.append(y_current)

rnn = RNN(time_step=5, input_size=4, hidden_size=3, output_size=2)

X = np.ones((5,4))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  x_, h_, y_, i_=sess.run([rnn.x, rnn.h,rnn.y,rnn.i], feed_dict={rnn.x:X})

  print('\n',x_)

  print('\n',h_)

  print('\n',y_)

  print('\n',i_)
