from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt

class TestModel_LSTM():
    def __init__(self, name, batch_size = 10, time_step = 28, input_dim = 28, learning_rate = 0.001):
        self.vocab_size = 10000
        self.timesteps = time_step
        self.num_hidden = 64
        self.batch_size = batch_size
        self.input_dim = input_dim

        self.X = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.timesteps, self.input_dim])

        self.x = tf.unstack(self.X, self.timesteps, 1)   #split embd_vec into 20 pieces, each piece has 256 inputs


        with tf.variable_scope(name):
            self.lstm_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)

            self.outputs, self.states = rnn.static_rnn(self.lstm_cell, self.x, dtype=tf.float32)

        self.output_reshape = tf.reshape(tf.stack(axis=1, values=self.outputs), [-1, self.num_hidden])
        # the sentence above can convert tensors with shape (x,y,z) to tensors with shape (x*y,z)
        self.W2 = tf.Variable(tf.random_normal([self.num_hidden, self.input_dim]))
        self.b2 = tf.Variable(tf.random_normal([self.input_dim]))
        self.logits = tf.matmul(self.output_reshape, self.W2) + self.b2

        self.logits = tf.reshape(self.logits, [self.batch_size, self.timesteps, self.input_dim])

        self.y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.timesteps, self.input_dim])
        self.loss = tf.losses.mean_squared_error(labels = self.logits, predictions= self.y)
        self.cost = tf.reduce_sum(self.loss)
        self.cost_test = tf.reduce_mean(self.loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        #
        # self.y = tf.placeholder("float", [None, self.num_classes])
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        #
        # self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


def main():
    #train_data, valid_data, test_data, voc = reader.ptb_raw_data('data/ptb.train.txt','data/ptb.valid.txt','data/ptb.test.txt')

    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    epoch_train = 20
    batch_size = 10
    time_step = 28
    input_dim =28
    simpleLSTM = TestModel_LSTM('mod1', batch_size = batch_size, time_step=time_step, input_dim = input_dim)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in np.arange(epoch_train):        #loop over all batches
            print('epoch:',i)
            n_batch = int(60000/batch_size)

            epoch_loss = 0
            for j in np.arange(n_batch):

                X_train_batch = x_train[j*batch_size:(j+1)*batch_size,:,:]

                _, cost, states = sess.run([simpleLSTM.optimizer, simpleLSTM.cost, simpleLSTM.states], feed_dict={simpleLSTM.X:X_train_batch, simpleLSTM.y:X_train_batch})
                epoch_loss+=cost
                if j%600==0:
                    print('training loss:', epoch_loss/(j*10+10))
                    test_index = int(4900*np.random.rand())
                    x_test_batch = x_test[test_index:test_index+10,:,:]
                    predictions, = sess.run([simpleLSTM.logits], feed_dict={simpleLSTM.X:x_test_batch, simpleLSTM.y:x_test_batch})
                    plt.figure()
                    for k in np.arange(10):
                        plt.subplot(2,5,k+1)
                        plt.imshow(predictions[k,:,:])
                    plt.show()


            n_batch_test = 1#5000/batch_size

            total_loss = 0
            for k in np.arange(n_batch_test):
                test_index = int(4000*np.random.rand())
                X_test_batch = x_test[test_index:test_index+10,:,:]
                #X_test_batch = x_test[k*batch_size:(k+1)*batch_size,:,:]

                test_cost, preds,= sess.run([simpleLSTM.cost_test, simpleLSTM.logits], feed_dict={simpleLSTM.X:X_test_batch, simpleLSTM.y:X_test_batch})
                total_loss+=test_cost

                plt.figure()
                plt.subplot(121)
                plt.imshow(X_test_batch[0,:,:])
                plt.subplot(122)
                plt.imshow(preds[0,:,:])
                plt.show()

            print('test loss:', total_loss/n_batch_test)



        sess.close()

main()
