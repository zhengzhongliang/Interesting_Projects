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

    self.y_tensor = tf.reshape(tf.stack(self.y), shape=[time_step, output_size])
    print('y shape:',self.y)
    print('y_tensor shape:',self.y_tensor)


    self.loss = tf.losses.mean_squared_error(labels=self.y_, predictions=self.y_tensor)

    self.tvars=[self.U, self.V, self.W]
    self.grads = tf.gradients(self.loss, self.tvars)

    optimizer = tf.train.GradientDescentOptimizer(0.1)    
    self.train_op = optimizer.apply_gradients(zip(self.grads, self.tvars))

rnn = RNN(time_step=5, input_size=4, hidden_size=30, output_size=2)

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

