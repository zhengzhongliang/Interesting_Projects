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

    i = tf.constant(1) 

    def cond_body_0(i, x,y,h, U , V, W):
      h_current = tf.sigmoid(tf.matmul(tf.reshape(x[i,:],shape=[1, input_size]),U))  # h_current(1,3)
      y_current = tf.matmul(h_current, V)    # y_current(1,2)
      h=tf.assign(h[i,:],tf.reshape(h_current,shape=[hidden_size]))
      y=tf.assign(y[i,:],tf.reshape(y_current,shape=[output_size]))

      #print(len((h,y)))

      return h,y


    def cond_body_1(i, x,y,h, U , V, W):
      h_prev = tf.reshape(h[i-1,:], shape=[1, hidden_size])
      print('h_prev:',h_prev)
      x_current= tf.reshape(x[i,:],shape=[1, input_size])
      print('x_current:',x_current)
      h_current = tf.sigmoid(tf.add(tf.matmul(x_current,U), tf.matmul(h_prev, W)))
      print('h_current:',h_current)
      y_current = tf.matmul(h_current, V)
      print('y_current:',y_current)
      h=tf.assign(h[i,:],tf.reshape(h_current,shape=[hidden_size]))
      y=tf.assign(y[i,:],tf.reshape(y_current,shape=[output_size]))

      #print(len((h,y)))

      return h,y

    self.h,self.y=tf.cond(tf.equal(i,tf.constant(0)), lambda:cond_body_0(i, self.x,self.y,self.h, self.U , self.V, self.W), lambda:cond_body_1(i, self.x,self.y,self.h, self.U , self.V, self.W))


rnn = RNN(time_step=5, input_size=4, hidden_size=3, output_size=2)

X = np.ones((5,4))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(sess.run([rnn.h,rnn.y], feed_dict={rnn.x:X}))
