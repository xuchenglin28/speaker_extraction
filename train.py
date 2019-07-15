#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017   Xu Chenglin(NTU, Singapore)
# Updated by Chenglin, Dec 2018, Jul 2019

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
from datetime import datetime
import pprint

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from model.model import Model
from utils.paddedFIFO_batch import paddedFIFO_batch
from utils.read_list import read_list

FLAGS = None

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    sys.stdout.flush()

def run_one_epoch(sess, coord, model, num_batches, is_eval):
    #Train/Eval one epoch of the model on the given data.
    loss = 0.0
    for batch in xrange(num_batches):
        if coord.should_stop():
            break
        if is_eval:
            loss_batch = sess.run(model._loss)
        else:
            _, loss_batch = sess.run([model._train_op, model._loss])
        loss += loss_batch

        if (batch+1) % 100 == 0:
            if is_eval:
                tf.logging.info("BATCH %d: EVAL AVG.LOSS %.4f at %s" % (batch+1, loss/(batch+1), datetime.now()))
            else:
                lr = sess.run(model._lr)
                tf.logging.info("BATCH %d: TRAIN AVG.LOSS %.4f with a learning rate %e at %s" % (batch+1, loss/(batch+1), lr, datetime.now()))
    loss /= num_batches

    return loss

def train():
    tr_tfrecords_list, tr_num_batches = read_list(FLAGS.lists_dir, "tr", FLAGS.batch_size)
    val_tfrecords_list, val_num_batches = read_list(FLAGS.lists_dir, "cv", FLAGS.batch_size)
  
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            with tf.name_scope('input'):
                cmvn = np.load(FLAGS.inputs_cmvn)
                cmvn_aux = np.load(FLAGS.inputs_cmvn.replace('cmvn', 'cmvn_aux'))
                tr_inputs, tr_inputs_cmvn, tr_inputs_cmvn_aux, tr_labels, tr_lengths, tr_lengths_aux = paddedFIFO_batch(tr_tfrecords_list, FLAGS.batch_size,
                    FLAGS.input_size, FLAGS.output_size, cmvn=cmvn, cmvn_aux=cmvn_aux, with_labels=FLAGS.with_labels, 
                    num_enqueuing_threads=FLAGS.num_threads, num_epochs=FLAGS.max_epochs, shuffle=FLAGS.shuffle)
                val_inputs, val_inputs_cmvn, val_inputs_cmvn_aux, val_labels, val_lengths, val_lengths_aux = paddedFIFO_batch(val_tfrecords_list, FLAGS.batch_size,
                    FLAGS.input_size, FLAGS.output_size, cmvn=cmvn, cmvn_aux=cmvn_aux, with_labels=FLAGS.with_labels,
                    num_enqueuing_threads=FLAGS.num_threads, num_epochs=FLAGS.max_epochs+1, shuffle=False)

        with tf.name_scope('model'):
            tr_model = Model(FLAGS, tr_inputs, tr_inputs_cmvn, tr_inputs_cmvn_aux, tr_labels, tr_lengths, tr_lengths_aux, infer=False)
            tf.get_variable_scope().reuse_variables()
            val_model = Model(FLAGS, val_inputs, val_inputs_cmvn, val_inputs_cmvn_aux, val_labels, val_lengths, val_lengths_aux, infer=False)

        show_all_variables()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        sess.run(init)

        checkpoint = tf.train.get_checkpoint_state(FLAGS.save_model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            tf.logging.info("Restore last saved model from " + checkpoint.model_checkpoint_path)
            tr_model.saver.restore(sess, checkpoint.model_checkpoint_path)
            best_model_path = checkpoint.model_checkpoint_path

            iter_idx = best_model_path.find('iter')
            mark_idx = best_model_path.find('_', iter_idx)
            start_iter = int(float(best_model_path[iter_idx+4:mark_idx]))

            lr_idx = best_model_path.find('lr')
            lr_mark_idx = best_model_path.find('_', lr_idx)
            FLAGS.learning_rate = float(best_model_path[lr_idx+2:lr_mark_idx])

            cv_idx = best_model_path.find('cv')
            cv_restore = float(best_model_path[cv_idx+2:])
            resume_training = 1
        else:
            start_iter = 0
            resume_training = 0
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            # Eval the init model to obtain the init loss on the development set before training.
            if resume_training:
                prev_val_loss = cv_restore
            else:
                prev_val_loss = run_one_epoch(sess, coord, val_model, val_num_batches, is_eval=True)
            tf.logging.info("INIT EVAL AVG.LOSS %.4f at %s" % (prev_val_loss, datetime.now()))

            sess.run(tf.assign(tr_model._lr, FLAGS.learning_rate))
            for epoch in xrange(start_iter, FLAGS.max_epochs):
                start_time = time.time()
                
                # Training for one epoch
                tr_loss = run_one_epoch(sess, coord, tr_model, tr_num_batches, is_eval=False)
                
                # Evaluating on the development set using the updated model
                val_loss = run_one_epoch(sess, coord, val_model, val_num_batches, is_eval=True)
                end_time = time.time()

                model_name = "nnet_iter%d_lr%e_tr%.4f_cv%.4f" % (epoch+1, FLAGS.learning_rate, tr_loss, val_loss)
                model_path = os.path.join(FLAGS.save_model_dir, model_name)
                rel_impr = (prev_val_loss - val_loss) / prev_val_loss
                if val_loss < prev_val_loss:
                    tr_model.saver.save(sess, model_path)
                    prev_val_loss = val_loss
                    best_model_path = model_path
                    tf.logging.info("ITER %d: TRAIN AVG.LOSS %.4f, CROSSVAL AVG.LOSS %.4f, LR %e, %s, %s USED TIME: %.2fs" % (
                        epoch+1, tr_loss, val_loss, FLAGS.learning_rate, "ACCEPTED", model_name, end_time-start_time))
                else:
                    tf.logging.info("ITER %d: TRAIN AVG.LOSS %.4f, CROSSVAL AVG.LOSS %.4f, LR %e, %s, %s USED TIME: %.2fs" % (
                        epoch+1, tr_loss, val_loss, FLAGS.learning_rate, "REJECTED", model_name, end_time-start_time))
                    tr_model.saver.restore(sess, best_model_path)

                # Reduce the learning rate when the relative improvement is lower than the threshold
                if rel_impr < FLAGS.reduce_lr_threshold:
                    FLAGS.learning_rate *= FLAGS.lr_reduction_factor
                    sess.run(tf.assign(tr_model._lr, FLAGS.learning_rate))

                # Stop the training when the relative improvement is lower than the threshold
                if rel_impr < FLAGS.stop_threshold:
                    if epoch+1 < FLAGS.min_epochs:
                        tf.logging.info("Continue to train the model with a minimum epoch of %d" % FLAGS.min_epochs)
                        continue
                    else:
                        tf.logging.info("The relative improvement is lower than the stopping threshold, stop training at a relative improvement of %f" % rel_impr)
                        break
        except Exception, e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)
        sess.close()

def main(_):
    if not os.path.exists(FLAGS.save_model_dir):
        os.makedirs(FLAGS.save_model_dir)
    train()

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lists_dir',
        type=str,
        default='tmp/',
        help="List to show where the data is."
    )
    parser.add_argument(
        '--with_labels',
        type=int,
        default=1,
        help='Whether the clean labels are included in the tfrecords.'
    )
    parser.add_argument(
        '--inputs_cmvn',
        type=str,
        default='tfrecords/tr_cmvn.npz',
        help="The global cmvn to normalize the inputs."
    )
    parser.add_argument(
        '--input_size',
        type=int,
        default=129,
        help="Input feature dimension (default 129 for 8kHz sampling rate)."
    )
    parser.add_argument(
        '--output_size',
        type=int,
        default=129,
        help="Output dimension (mask dimension, default 129 for 8kHz sampling rate)."
    )
    parser.add_argument(
        '--aux_hidden_size',
        type=int,
        default=512,
        help="The dimension of hidden layer in auxillary network."
    )
    parser.add_argument(
        '--aux_output_size',
        type=int,
        default=30,
        help="The dimension of output layer in auxillary network, equals to the number of sub-layers in the adapt layer."
    )
    parser.add_argument(
        '--rnn_size',
        type=int,
        default=896,
        help="Number of units in a rnn layer."
    )
    parser.add_argument(
        '--rnn_num_layers',
        type=int,
        default=3,
        help="Number of rnn layers."
    )
    parser.add_argument(
        '--mask_type',
        type=str,
        default='relu',
        help="Mask avtivation funciton, now only support sigmoid or relu"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help="Minibatch size."
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0005,
        help="Initial learning rate."
    )
    parser.add_argument(
        '--min_epochs',
        type=int,
        default=30,
        help="Minimum epochs when training the model."
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=100,
        help="Maximum epochs when training the model."
    )
    parser.add_argument(
        '--lr_reduction_factor',
        type=float,
        default=0.5,
        help="Factor for reducing the learning rate."
    )
    parser.add_argument(
        '--reduce_lr_threshold',
        type=float,
        default=0.0,
        help="Threshold to decide when to reduce the learning rate."
    )
    parser.add_argument(
        '--stop_threshold',
        type=float,
        default=0.0001,
        help="Threshold to decide when to stop training."
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=12,
        help='Number of threads for paralleling.'
    )
    parser.add_argument(
        '--save_model_dir',
        type=str,
        default='exp/model_name',
        help="Directory to save the training model in every epoch."
    )
    parser.add_argument(
        '--keep_prob',
        type=float,
        default=0.7,
        help="Keep probability for training with a dropout (default: 1-dropout_rate)."
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=5.0,
        help="The max gradient normalization."
    )
    parser.add_argument(
        '--del_factor',
        type=float,
        default=0.0,
        help="weight for delta objective function, if larger than 0, delta objective function is applied."
    )
    parser.add_argument(
        '--acc_factor',
        type=float,
        default=0.0,
        help="weight for acceleration objective function, if larger than 0, delta objective function is applied."
    )
    parser.add_argument(
        '--dynamic_win',
        type=int,
        default=2,
        help="window size of the order in calculation of dynamic objective functions, default is 2."
    )
    parser.add_argument(
        '--mag_factor',
        type=float,
        default=0.0,
        help="weight for static (i.e., magnitude) objective function, if larger than 0, static objective function is applied."
    )
    parser.add_argument(
        '--shuffle',
        type=int,
        default=0,
        help="Whether shuffle data."
    )
    parser.add_argument(
        '--power_num',
        type=int,
        default=2,
        help="The power to calculate the loss, if set to 2, it's squared L2, if set to 1, it's L1."
    )
    FLAGS, unparsed = parser.parse_known_args()
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__dict__)
    sys.stdout.flush()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
