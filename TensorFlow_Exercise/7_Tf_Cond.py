import tensorflow as tf

x = tf.get_variable(name='x', initializer=tf.constant(5))
y = tf.get_variable(name='y', initializer=tf.constant(6))

def function_1(x,y):
  return tf.add(x,y),x

def function_2(x,y):
  return tf.subtract(x,y),x

a,x = tf.cond(tf.less(x,y), lambda: function_1(x,y), lambda:function_2(x,y))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(sess.run(a))
