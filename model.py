#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class Model():
    def __init__(self, learning_rate=0.001, batch_size=16, num_steps=32, num_words=5000, dim_embedding=128, rnn_layers=3):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_words = num_words
        self.dim_embedding = dim_embedding
        self.rnn_layers = rnn_layers
        self.learning_rate = learning_rate

    def build(self, embedding_file=None, is_train=False):
        # global step
        self.global_step = tf.Variable(
            0, trainable=False, name='self.global_step', dtype=tf.int64)

        self.X = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='input')
        self.Y = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='label')

        self.keep_prob = tf.placeholder(tf.float32, name='self.keep_prob')

        with tf.variable_scope('embedding'):
            if embedding_file:
                # if embedding file provided, use it.
                embedding = np.load(embedding_file)
                embed = tf.constant(embedding, name='embedding')
            else:
                # if not, initialize an embedding and train it.
                embed = tf.get_variable(
                    'embedding', [self.num_words, self.dim_embedding])
                tf.summary.histogram('embed', embed)

            data = tf.nn.embedding_lookup(embed, self.X)

        with tf.variable_scope('rnn'):
            ##################
            # Your Code here
            ##################
            #cell = tf.nn.rnn_cell.MultiRNNCell([self.get_rnn_cell(128, self.keep_prob, is_train)] * self.rnn_layers, state_is_tuple=False)
            cell = tf.nn.rnn_cell.BasicRNNCell(128)
            init_state = cell.zero_state(tf.shape(self.X)[0], dtype=tf.float32)
            outputs_tensor = []
            """
            for time_step in range(30):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                    cell_output, init_state = cell(data[:, time_step, :], init_state)
                    outputs_tensor.append(cell_output)
            """
            init_state = tf.zeros([self.batch_size, 128])
            self.state_tensor = init_state
            outputs_tensor, final_state = tf.nn.dynamic_rnn(cell, data, initial_state=init_state)  
            self.outputs_state_tensor = final_state

        # concate every time step
        seq_output = tf.concat(outputs_tensor, 1)

        # flatten it
        seq_output_final = tf.reshape(seq_output, [-1, self.dim_embedding])

        with tf.variable_scope('softmax'):
            ##################
            # Your Code here
            ##################
            W = tf.get_variable('W', [self.dim_embedding, self.num_words])
            b = tf.get_variable('b', [self.num_words], initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(seq_output_final, W) + b

        tf.summary.histogram('logits', logits)

        self.predictions = tf.nn.softmax(logits, name='predictions')

        y_one_hot = tf.one_hot(self.Y, self.num_words)
        y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y_reshaped)
        self.loss = tf.reduce_mean(loss)

        # gradient clip
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(
            zip(grads, tvars), global_step=self.global_step)

        tf.summary.scalar('loss', self.loss)

        self.merged_summary_op = tf.summary.merge_all()
        
    def get_rnn_cell(self, hidden_state=128, keep_prob=0.8, is_train=False):
        """"""
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_state, reuse = True, state_is_tuple=False)
        if(True == is_train):
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        return cell

