import tensorflow as tf
import numpy as  np

#The level of tensorflow: Graph, Variable Scope, Variable, Name Scope.

# Variable Scope and Varibles are like pointers, while Name Scope is like variable name in python. 

g1 = tf.Graph()                    # g1 is a graph object
with g1.as_default() as g:
  with tf.name_scope(name='g1') as scope:
    m1 = tf.constant([[3., 2.]], name='m1')
    m2 = tf.constant([[3.],[3.]], name='m2')
    m3 = tf.matmul(m1, m2, name='m3')
tf.reset_default_graph()

g2 = tf.Graph()                    # g2 is a graph object
with g2.as_default() as g:           #as_default shows that we are currently adding components to g2.
  with tf.name_scope(name='g2') as scope:
    m4 = tf.constant([[4., 2.]], name='m4')
    m5 = tf.constant([[4.],[4.]], name='m5')
    m6 = tf.matmul(m4, m5, name='m6')
tf.reset_default_graph()             #

#with tf.Session(graph=g1) as sess:
#  print(sess.run(m3))    #works fine


# in tensorflow there are two ways of accessing the variables. One is by the variable name, this is equivalent to access the variable directly by acessing the storage in memory. Another way is by "names" and "name scopes". The names are like pointer variables. We can access the storage in memory by calling these pointer variables. Note that different pointer variables may refer to the same piece of storage in memory.

# some useful APIs:
# sess.graph.get_tensor_by_name('name')
# sess.graph.get_operation()

with tf.Session(graph=g1) as sess:
  print([op.name for op in sess.graph.get_operations()])   # this will list all operations in a graph
  print([op.values() for op in sess.graph.get_operations()])   # this will list all tensors in a graph
  print(sess.graph.get_tensor_by_name('g1/m3:0'))
