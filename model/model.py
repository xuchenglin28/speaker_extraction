#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017  Xu Chenglin(NTU, Singapore)
# Updated by Chenglin, Dec 2018

"""
1. Build speech separation network structure
2. Calculate the objective loss
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn
from utils.comp_dynamic_feature import comp_dynamic_feature

class Model(object):

    def __init__(self, config, inputs, inputs_norm, inputs_norm_aux, labels=None, lengths=None, lengths_aux=None, infer=False):
        self._config = config
        self._mixed = inputs
        self._inputs = inputs_norm
        self._inputs_aux = inputs_norm_aux
        if labels is not None:
            self._labels = labels
        self._lengths = lengths
        self._lengths_aux = lengths_aux
        self._infer = infer

        self.build_model()

    def build_model(self):
        self.build_net()
        if self._infer: return
        self.cal_loss()
        if tf.get_variable_scope().reuse: return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars), self._config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    # build auxiliary network, since this speaker embedding network can be used on top of extected speech for speaker verification
    # we are going to share weights
    def build_net_aux(self, inputs, lengths):
        outputs = tf.reshape(inputs, [self._config.batch_size, -1, self._config.input_size])
        # BLSTM layer
        with tf.variable_scope('blstm_aux'):
            def lstm_cell():
                if not self._infer and self._config.keep_prob < 1.0:
                    return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self._config.aux_hidden_size), output_keep_prob=self._config.keep_prob)
                else:
                    return tf.contrib.rnn.BasicLSTMCell(self._config.aux_hidden_size)

            # tf.nn.rnn_cell.MultiRNNCell in r1.12
            lstm_fw_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self._config.rnn_num_layers)], state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self._config.rnn_num_layers)], state_is_tuple=True)
            lstm_fw_cell = self._unpack_cell(lstm_fw_cell)
            lstm_bw_cell = self._unpack_cell(lstm_bw_cell)
            outputs, fw_final_states, bw_final_states = rnn.stack_bidirectional_dynamic_rnn(cells_fw=lstm_fw_cell, cells_bw=lstm_bw_cell, inputs=outputs, dtype=tf.float32,
                         sequence_length=lengths)
            outputs = tf.reshape(outputs, [-1, 2*self._config.aux_hidden_size]) # transform blstm outputs into right output size

        with tf.variable_scope('layer2_aux'):
            weights2, biases2 = self._weight_and_bias(2*self._config.aux_hidden_size, self._config.aux_hidden_size)
            outputs = tf.nn.relu(tf.matmul(outputs, weights2) +  biases2)

        with tf.variable_scope('layer3_aux'):
            weights3, biases3 = self._weight_and_bias(self._config.aux_hidden_size, self._config.aux_output_size)
            outputs = tf.matmul(outputs, weights3) + biases3
            outputs = tf.reshape(outputs, [self._config.batch_size, -1, self._config.aux_output_size])
            # average over the frames to get the speaker embedding
            spk_embed = tf.reduce_sum(outputs, 1)/tf.reshape(tf.to_float(self._lengths_aux), (-1,1))

        return spk_embed

    def build_net(self):

        # build auxiliary network to get the speaker embedding used for speaker extraction network
        with tf.variable_scope('spk_embed') as scope:
            spk_embed_aux = self.build_net_aux(self._inputs_aux, self._lengths_aux)

        outputs = tf.reshape(self._inputs, [self._config.batch_size, -1, self._config.input_size])        
        # BLSTM layer
        with tf.variable_scope('blstm'):
            def lstm_cell():
                if not self._infer and self._config.keep_prob < 1.0:
                    return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self._config.rnn_size), output_keep_prob=self._config.keep_prob)
                else:
                    return tf.contrib.rnn.BasicLSTMCell(self._config.rnn_size)

            # tf.nn.rnn_cell.MultiRNNCell in r1.12
            lstm_fw_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self._config.rnn_num_layers)], state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self._config.rnn_num_layers)], state_is_tuple=True)
            lstm_fw_cell = self._unpack_cell(lstm_fw_cell)
            lstm_bw_cell = self._unpack_cell(lstm_bw_cell)
            outputs, fw_final_states, bw_final_states = rnn.stack_bidirectional_dynamic_rnn(cells_fw=lstm_fw_cell, cells_bw=lstm_bw_cell, inputs=outputs, dtype=tf.float32,
                         sequence_length=self._lengths)

        # speaker adaptation layer by concat the output from auxiliary network
        with tf.variable_scope('adapt_concat'):
            outputs = tf.reshape(outputs, [self._config.batch_size, -1, 2*self._config.rnn_size])
            frame_num = tf.shape(outputs)[1]
            spk_embed = tf.transpose(tf.reshape(tf.tile(tf.reshape(spk_embed_aux, (-1, 1)), (frame_num, 1)), (frame_num, self._config.batch_size, self._config.aux_output_size)), perm=[1,0,2])

            outputs = tf.concat([outputs, spk_embed], 2)
           
            # remove the part out of the lenghts when concate speaker embeddings
            outputs = tf.multiply(tf.expand_dims(tf.sequence_mask(self._lengths, dtype=tf.float32), -1), outputs)
            concat_dim = 2*self._config.rnn_size+self._config.aux_output_size

        outputs = tf.reshape(outputs, [-1, concat_dim])

            
        # one more fully connected layer
        with tf.variable_scope('fc1'):
            weights1, biases1 = self._weight_and_bias(concat_dim, self._config.rnn_size)
            outputs = tf.nn.relu(tf.matmul(outputs, weights1) + biases1)

            outputs = tf.reshape(outputs, [self._config.batch_size, -1, self._config.rnn_size])

        # BLSTM layer
        with tf.variable_scope('blstm2'):
            def lstm_cell():
                if not self._infer and self._config.keep_prob < 1.0:
                    return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self._config.rnn_size), output_keep_prob=self._config.keep_prob)
                else:
                    return tf.contrib.rnn.BasicLSTMCell(self._config.rnn_size)

            # tf.nn.rnn_cell.MultiRNNCell in r1.12
            lstm_fw_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self._config.rnn_num_layers)], state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self._config.rnn_num_layers)], state_is_tuple=True)
            lstm_fw_cell = self._unpack_cell(lstm_fw_cell)
            lstm_bw_cell = self._unpack_cell(lstm_bw_cell)
            outputs, fw_final_states, bw_final_states = rnn.stack_bidirectional_dynamic_rnn(cells_fw=lstm_fw_cell, cells_bw=lstm_bw_cell, inputs=outputs, dtype=tf.float32,
                         sequence_length=self._lengths)
            outputs = tf.reshape(outputs, [-1, 2*self._config.rnn_size])
        
        # one more fully connected layer
        with tf.variable_scope('fc2'):
            weights2, biases2 = self._weight_and_bias(2*self._config.rnn_size, self._config.rnn_size)
            outputs = tf.nn.relu(tf.matmul(outputs, weights2) + biases2)

        # Mask estimation layer
        with tf.variable_scope('mask'):
            weights_m, biases_m = self._weight_and_bias(self._config.rnn_size, self._config.output_size)
            if self._config.mask_type.lower() == 'relu':
                mask = tf.nn.relu(tf.matmul(outputs, weights_m) + biases_m)
            else:
                mask = tf.nn.sigmoid(tf.matmul(outputs, weights_m) + biases_m)
            
            self._mask = tf.reshape(mask, [self._config.batch_size, -1, self._config.output_size])

            self._sep = self._mask * self._mixed

        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=50)

    def cal_loss(self):
        if self._config.mag_factor > 0.0:
            cost = tf.reduce_sum(tf.reduce_sum(tf.abs(tf.pow(self._sep - self._labels, self._config.power_num)), 1), 1)
            #cost = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.pow(self._sep - self._labels, self._config.power_num)), 1), 1)
            cost = tf.multiply(self._config.mag_factor, cost)
        else:
            cost = 0.0

        if self._config.del_factor > 0.0:
            sep_delta = comp_dynamic_feature(self._sep, self._config.dynamic_win, self._config.batch_size, self._lengths)
            labels_delta = comp_dynamic_feature(self._labels, self._config.dynamic_win, self._config.batch_size, self._lengths)
            cost_del = tf.reduce_sum(tf.reduce_sum(tf.abs(tf.pow(sep_delta - labels_delta, self._config.power_num)), 1), 1)
            #cost_del = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.pow(sep_delta - labels_delta, self._config.power_num)), 1), 1)
            cost += tf.multiply(self._config.del_factor, cost_del)

        if self._config.acc_factor > 0.0:
            sep_acc = comp_dynamic_feature(sep_delta, self._config.dynamic_win, self._config.batch_size, self._lengths)
            labels_acc = comp_dynamic_feature(labels_delta, self._config.dynamic_win, self._config.batch_size, self._lengths)
            cost_acc = tf.reduce_sum(tf.reduce_sum(tf.abs(tf.pow(sep_acc - labels_acc, self._config.power_num)), 1), 1)
            #cost_acc = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.pow(sep_acc - labels_acc, self._config.power_num)), 1), 1)
            cost += tf.multiply(self._config.acc_factor, cost_acc)

        self._loss = tf.reduce_mean(tf.div(cost, tf.to_float(self._lengths)))

    def _weight_and_bias(self, input_size, output_size):
        weights = tf.get_variable('weights', [input_size, output_size], initializer=tf.random_normal_initializer(stddev=0.01))
        biases = tf.get_variable('biases', [output_size], initializer=tf.constant_initializer(0.0))
        return weights, biases

    def _unpack_cell(self, cell):
        if isinstance(cell,tf.contrib.rnn.MultiRNNCell):
            return cell._cells
        else:
            return [cell]
