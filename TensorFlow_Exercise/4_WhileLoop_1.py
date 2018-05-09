import tensorflow as tf
import numpy as np
import collections

#logistic = lambda x: 1.0/(1+np.exp(-x))
#print(logistic(3))


# Example 1: 
#j = tf.constant(0)
#c = lambda i: tf.less(i, 9)   # loop termination condition
#b = lambda i: tf.add(i, 1)    # loop body: loop body that is executed in each loop
#r = tf.while_loop(c, b, [j])  # [] includes the initial condition of loop

#with tf.Session() as sess:
#  print(sess.run(r))

    # How to understand this: in each loop, the value of j will be passed to c and b. 
    # this loop function can be unfolded by:
    # j
    # while (c(j)):
    #   j=b(j)

    # question: what is the body is very complicated?
    # see here: https://stackoverflow.com/questions/47955437/tensorflow-stacking-tensors-in-while-loop
    # the loop body can be defined as a function
    # the resulted code is shown in example 3.



# Example 2:

#Pair = collections.namedtuple('Pair', 'j, k')
#ijk_0 = (tf.constant(0), Pair(tf.constant(1), tf.constant(2)))
#c = lambda i, p: i < 10                            
#b = lambda i, p: (i + 1, Pair((p.j + p.k), (p.j - p.k)))
#ijk_final = tf.while_loop(c, b, ijk_0)

#with tf.Session() as sess:
#  print(sess.run(ijk_final))

# How to understand the code:
# the loop body is a tuple made by all relevant variables.


# Example 3:
def body(i,p):
  return (i + 1, Pair((p.j + p.k), (p.j - p.k)))

Pair = collections.namedtuple('Pair', 'j, k')
ijk_0 = (tf.constant(0), Pair(tf.constant(1), tf.constant(2)))
c = lambda i, p: i < 10                            
b = lambda i, p: body(i,p)
ijk_final = tf.while_loop(c, b, ijk_0)

with tf.Session() as sess:
  print(sess.run(ijk_final))














