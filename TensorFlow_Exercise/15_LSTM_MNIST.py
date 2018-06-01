import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class LSTM():
    
  def __init__(self, time_step, input_size, hidden_size, output_size):
    self.x = tf.placeholder(shape=[time_step, input_size] ,dtype=tf.float32)
    self.y_ = tf.placeholder(shape=[time_step, output_size] ,dtype=tf.float32)
    self.h = list()
    self.y = list()
    self.c = list()

    
    self.V = tf.get_variable(name='V',shape=[hidden_size, output_size],dtype=tf.float32)
    self.W_g = tf.get_variable(name='W_g',shape=[hidden_size, hidden_size],dtype=tf.float32)
    self.W_i = tf.get_variable(name='W_i',shape=[hidden_size, hidden_size],dtype=tf.float32)
    self.W_f = tf.get_variable(name='W_f',shape=[hidden_size, hidden_size],dtype=tf.float32)
    self.W_o = tf.get_variable(name='W_o',shape=[hidden_size, hidden_size],dtype=tf.float32)
    self.U_g = tf.get_variable(name='U_g',shape=[input_size, hidden_size],dtype=tf.float32)   # Change Initilizer Here!
    self.U_i = tf.get_variable(name='U_i',shape=[input_size, hidden_size],dtype=tf.float32)
    self.U_f = tf.get_variable(name='U_f',shape=[input_size, hidden_size],dtype=tf.float32)
    self.U_o = tf.get_variable(name='U_o',shape=[input_size, hidden_size],dtype=tf.float32)

    self.i = tf.constant(0) 

    for i in np.arange(time_step):
      if(i==0):
        x_current = tf.reshape(self.x[i,:],shape=[1, input_size])
        g = tf.tanh(tf.matmul(x_current,self.U_g))
        i = tf.sigmoid(tf.matmul(x_current,self.U_i))
        f = tf.sigmoid(tf.matmul(x_current,self.U_f))
        o = tf.sigmoid(tf.matmul(x_current,self.U_o))
        c_current = tf.multiply(g,i)
        h_current = tf.multiply(o,tf.tanh(c_current))
        y_current = tf.matmul(h_current, self.V)    # y_current(1,2)
        self.c.append(c_current)
        self.h.append(h_current)
        self.y.append(y_current)
      else:
        c_prev = tf.reshape(self.c[i-1], shape=[1, hidden_size])
        h_prev = tf.reshape(self.h[i-1],shape=[1, hidden_size])
        x_current = tf.reshape(self.x[i,:],shape=[1, input_size])
        g = tf.tanh(tf.add(tf.matmul(x_current,self.U_g), tf.matmul(h_prev,self.W_g)))
        i = tf.sigmoid(tf.add(tf.matmul(x_current,self.U_i), tf.matmul(h_prev,self.W_i)))
        f = tf.sigmoid(tf.add(tf.matmul(x_current,self.U_f), tf.matmul(h_prev,self.W_f)))
        o = tf.sigmoid(tf.add(tf.matmul(x_current,self.U_o), tf.matmul(h_prev,self.W_o)))
        c_current = tf.add(tf.multiply(g,i), tf.multiply(f,c_prev))
        h_current = tf.multiply(o,tf.tanh(c_current))
        y_current = tf.matmul(h_current, self.V)    # y_current(1,2)
        self.c.append(c_current)
        self.h.append(h_current)
        self.y.append(y_current)

    self.y_tensor = tf.reshape(tf.stack(self.y), shape=[time_step, output_size])
    print('y shape:',self.y)
    print('y_tensor shape:',self.y_tensor)


    self.loss = tf.losses.mean_squared_error(labels=self.y_, predictions=self.y_tensor)

    self.tvars=[self.V, self.W_g, self.W_i,self.W_f,self.W_o,self.U_g,self.U_i,self.U_f,self.U_o]
    self.grads = tf.gradients(self.loss, self.tvars)

    optimizer = tf.train.GradientDescentOptimizer(0.1)    
    self.train_op = optimizer.apply_gradients(zip(self.grads, self.tvars))


rnn = LSTM(time_step=5, input_size=4, hidden_size=30, output_size=2)

X = np.ones((5,4))
y_ = np.ones((5,2))*0.3


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in np.arange(100000):
    _, loss_, y_pred=sess.run([rnn.train_op, rnn.loss, rnn.y], feed_dict={rnn.x:X, rnn.y_:y_})
    if i==0:
      print(y_pred)
      input('Press enter to continue')

    if((i+1)%1000==0):
      print('\n',loss_)
  print(y_pred)
