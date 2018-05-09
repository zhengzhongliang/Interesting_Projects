import tensorflow as tf
import numpy as np

#=======================================================================================
# Example 4: slice assignment using "tf.stack"
A = tf.placeholder(shape=[3],dtype=tf.float32)

C = tf.get_variable(name='C',initializer=tf.constant(np.ones((1,3))))

B = list()
B.append(A)
B.append(A)

D = tf.stack([A,A],0)    # this will change the dimension of B

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(sess.run([D], feed_dict={A:np.ones(3)}))


#=======================================================================================
# Example 3: slice assignment using "tf.concat""
#A = tf.placeholder(shape=[1,3],dtype=tf.float32)

#C = tf.get_variable(name='C',initializer=tf.constant(np.ones((1,3))))

#B = tf.concat([A,A],0)

#with tf.Session() as sess:
#  sess.run(tf.global_variables_initializer())
#  print(sess.run([B], feed_dict={A:np.ones((1,3))}))


#=======================================================================================
# Example 6: slice assignment, direct assignment, failed!
#A = tf.placeholder(shape=[1,3],dtype=tf.float32)

#C = tf.get_variable(name='C',initializer=tf.constant(np.ones((1,3))))

#B = tf.get_variable(name='B',shape=[3,3])    # this will change the dimension of B

#B[0,:]=A
#B[1,:]=A
#B[2,:]=A

#with tf.Session() as sess:
#  sess.run(tf.global_variables_initializer())
#  print(sess.run([B], feed_dict={A:np.ones((1,3))}))

#=======================================================================================
# Example 5: slice assignment using "tf.assign", failed!
#A = tf.placeholder(shape=[1,3],dtype=tf.float64)

#C = tf.get_variable(name='C',initializer=tf.constant(np.ones((1,3))), dtype=tf.float64)

#B = tf.get_variable(name='B',shape=[3,3], dtype=tf.float64)    # this will change the dimension of B

#B = B[0,:].assign(C)
#B = B[1,:].assign(C)
#B = B[2,:].assign(C)

#with tf.Session() as sess:
#  sess.run(tf.global_variables_initializer())
#  print(sess.run([B], feed_dict={A:np.ones((1,3))}))

#=======================================================================================
# Example 2: 

#A = tf.placeholder(shape=[3,3],dtype=tf.float32)

#B = A[0,:]

#with tf.Session() as sess:
#  print(sess.run(B, feed_dict={A:np.zeros((3,3))}))


#=======================================================================================
# Example 1:

#init = tf.constant(np.ones((3,3,3)))

#A = tf.get_variable(name='a',initializer=init)

#B = A[0,0,:]
#C = tf.gather(A,[1,2])

#with tf.Session() as sess:
#  sess.run(tf.global_variables_initializer())
#  print(sess.run([A,B,C]))
