import numpy as np
import tensorflow as tf

class RNN():
    
    


  def __init__(self, time_step, input_size, hidden_size, output_size):
    self.x = tf.placeholder(shape=[time_step, input_size] ,dtype=tf.float32)
    self.y_ = tf.placeholder(shape=[time_step, output_size] ,dtype=tf.float32)
    self.h = tf.get_variable(name='h', shape=[1, hidden_size],dtype=tf.float32)
    self.y = tf.get_variable(name='y', shape=[1, output_size],dtype=tf.float32)

    self.U = tf.get_variable(name='U',shape=[input_size, hidden_size],dtype=tf.float32)   # Change Initilizer Here!
    self.V = tf.get_variable(name='V',shape=[hidden_size, output_size],dtype=tf.float32)
    self.W = tf.get_variable(name='W',shape=[hidden_size, hidden_size],dtype=tf.float32)

    self.i = tf.constant(0) 

    def loop_body_def(i, x,y,h, U , V, W):
      def cond_body_0(i, x,y,h, U , V, W):
        h_current = tf.sigmoid(tf.matmul(tf.reshape(x[i,:],shape=[1, input_size]),U))
        y_current = tf.matmul(h_current, V)    # y_current(1,2)
        
        h=h_current
        y=y_current

        print('h shape:',h)
        print('y shape:',y)

        return h,y


      def cond_body_1(i, x,y,h, U , V, W):
        h_prev = tf.reshape(h[i-1,:], shape=[1, hidden_size])
        x_current= tf.reshape(x[i,:],shape=[1, input_size])
        h_current = tf.sigmoid(tf.add(tf.matmul(x_current,U), tf.matmul(h_prev, W)))
        y_current = tf.matmul(h_current, V)

        print('h shape:',h)
        print('y shape:',y)
        print('h_current shape:',h_current)
        print('y_current shape:',y_current)

        h=tf.concat([h,h_current],axis=0)
        y=tf.concat([y,y_current],axis=0)

        print('h shape:',h)
        print('y shape:',y)

        return h,y

      h,y=tf.cond(tf.equal(i,tf.constant(0)), lambda:cond_body_0(i, x,y,h, U , V, W), lambda:cond_body_1(i, x,y,h, U , V, W))

      return (tf.add(i,tf.constant(1)), x,y,h, U , V, W)
      

    def loop_cond_def(i, x,y,h, U , V, W):
      return tf.less(i, tf.constant(3))

    self.i,self.x, self.y, self.h,self.U, self.V, self.W=tf.while_loop(loop_cond_def, loop_body_def, [self.i,self.x, self.y, self.h,self.U, self.V, self.W])


rnn = RNN(time_step=5, input_size=4, hidden_size=3, output_size=2)

X = np.ones((5,4))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(sess.run([rnn.h,rnn.y,rnn.i], feed_dict={rnn.x:X}))
