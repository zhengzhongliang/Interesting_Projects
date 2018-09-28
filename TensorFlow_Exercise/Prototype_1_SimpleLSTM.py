from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time

import numpy as np
import tensorflow as tf
import reader
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

class TestModel_LSTM():
    def __init__(self, name, batch_size = 10, time_step = 20, learning_rate = 0.001):
        self.vocab_size = 10000
        self.timesteps = time_step
        self.num_hidden = 256
        self.batch_size = batch_size

        self.X = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.timesteps])
        word_embeddings = tf.get_variable('word_embeddings',[self.vocab_size, self.num_hidden])
        self.embd_vec = tf.nn.embedding_lookup(word_embeddings, self.X)     #shape of embd_vec: (?, 20, 256)

        self.x = tf.unstack(self.embd_vec, self.timesteps, 1)   #split embd_vec into 20 pieces, each piece has 256 inputs


        with tf.variable_scope(name):
            self.lstm_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)

            self.outputs, self.states = rnn.static_rnn(self.lstm_cell, self.x, dtype=tf.float32)

        self.output_reshape = tf.reshape(tf.stack(axis=1, values=self.outputs), [-1, self.num_hidden])
        # the sentence above can convert tensors with shape (x,y,z) to tensors with shape (x*y,z)
        self.W1 = tf.Variable(tf.random_normal([self.num_hidden, self.vocab_size]))
        self.b1 = tf.Variable(tf.random_normal([self.vocab_size]))
        self.logits = tf.matmul(self.output_reshape, self.W1) + self.b1

        self.logits = tf.reshape(self.logits, [self.batch_size, self.timesteps, self.vocab_size])

        self.y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.timesteps])
        self.loss = tf.contrib.seq2seq.sequence_loss(
            self.logits,
            self.y,
            tf.ones([self.batch_size, self.timesteps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True
        )
        self.cost = tf.reduce_sum(self.loss)
        self.cost_test = tf.reduce_mean(self.loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        #
        # self.y = tf.placeholder("float", [None, self.num_classes])
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        #
        # self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


def main():
    train_data, valid_data, test_data, voc = reader.ptb_raw_data('data/ptb.train.txt','data/ptb.valid.txt','data/ptb.test.txt')

    epoch_train = 10
    batch_size = 10
    time_step = 20
    simpleLSTM = TestModel_LSTM('mod1', batch_size = batch_size, time_step=time_step)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in np.arange(epoch_train):        #loop over all batches
            print('epoch:',i)
            n_batch = int((len(train_data)-1)/batch_size/time_step)
            n_word = time_step*batch_size    # number of words in each batch
            for j in np.arange(n_batch):
                X_train_batch = train_data[j*n_word:(j+1)*n_word]
                y_train_batch = train_data[j*n_word+1 : (j+1)*n_word+1]

                X_train_batch = np.array(X_train_batch).reshape((batch_size, time_step))
                y_train_batch = np.array(y_train_batch).reshape((batch_size, time_step))


                _, cost, states = sess.run([simpleLSTM.optimizer, simpleLSTM.cost, simpleLSTM.states], feed_dict={simpleLSTM.X:X_train_batch, simpleLSTM.y:y_train_batch})

                if j%100==0:
                    print('training loss:', cost/time_step)
            n_batch_test = int((len(test_data)-1)/batch_size/time_step)
            n_word = time_step*batch_size

            total_loss = 0
            for k in np.arange(n_batch_test):
                X_test_batch = test_data[k*n_word:(k+1)*n_word]
                y_test_batch = test_data[k * n_word + 1: (k + 1) * n_word + 1]

                X_test_batch = np.array(X_test_batch).reshape((batch_size, time_step))
                y_test_batch = np.array(y_test_batch).reshape((batch_size, time_step))

                test_cost, = sess.run([simpleLSTM.cost_test], feed_dict={simpleLSTM.X:X_test_batch, simpleLSTM.y:y_test_batch})
                total_loss+=test_cost

            print('test loss:', total_loss/n_batch_test)

        sess.close()

main()
