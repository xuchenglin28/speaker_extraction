#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017   Xu Chenglin(NTU, Singapore)
# Updated by Chenglin, Dec 2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os, sys, re, struct, time
import pprint
import numpy as np
import tensorflow as tf
from model.model import Model
from utils.paddedFIFO_batch import paddedFIFO_batch
from utils.read_list import read_list

from utils.audioread import audioread
from utils.sigproc import framesig,magspec,deframesig
from utils.normhamming import normhamming
import scipy.io.wavfile as wav

FLAGS = None

def reconstruct(enhan_spec, noisy_file):

    rate, sig, nb_bits = audioread(noisy_file)
    frames = framesig(sig, FLAGS.FFT_LEN, FLAGS.FRAME_SHIFT, lambda x: normhamming(x), True)
    phase_noisy, mag_noisy = magspec(frames, FLAGS.FFT_LEN)

    spec_comp = enhan_spec * np.exp(phase_noisy * 1j)
    enhan_frames = np.fft.irfft(spec_comp)
    enhan_sig = deframesig(enhan_frames, len(sig), FLAGS.FFT_LEN, FLAGS.FRAME_SHIFT, lambda x: normhamming(x))
    enhan_sig = enhan_sig / np.max(np.abs(enhan_sig)) * np.max(np.abs(sig))

    enhan_sig = np.round(enhan_sig * float(2 ** (nb_bits - 1)))
    if nb_bits == 16:
        enhan_sig = enhan_sig.astype(np.int16)
    elif nb_bits == 32:
        enhan_sig = enhan_sig.astype(np.int32)

    return enhan_sig, rate

def decode():
    tfrecords_list, num_batches = read_list(FLAGS.lists_dir, FLAGS.data_type, FLAGS.batch_size)

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            with tf.name_scope('input'):
                cmvn = np.load(FLAGS.inputs_cmvn)
                cmvn_aux = np.load(FLAGS.inputs_cmvn.replace('cmvn', 'cmvn_aux'))
                if FLAGS.with_labels:
                    inputs, inputs_cmvn, inputs_cmvn_aux, labels, lengths, lengths_aux = paddedFIFO_batch(tfrecords_list, FLAGS.batch_size,
                        FLAGS.input_size, FLAGS.output_size, cmvn=cmvn, cmvn_aux=cmvn_aux, with_labels=FLAGS.with_labels, 
                        num_enqueuing_threads=1, num_epochs=1, shuffle=False)
                else:
                    inputs, inputs_cmvn, inputs_cmvn_aux, lengths, lengths_aux = paddedFIFO_batch(tfrecords_list, FLAGS.batch_size,
                        FLAGS.input_size, FLAGS.output_size, cmvn=cmvn, cmvn_aux=cmvn_aux, with_labels=FLAGS.with_labels,
                        num_enqueuing_threads=1, num_epochs=1, shuffle=False)
                    labels = None
               
        with tf.name_scope('model'):
            model = Model(FLAGS, inputs, inputs_cmvn, inputs_cmvn_aux, labels, lengths, lengths_aux, infer=True)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init)

        checkpoint = tf.train.get_checkpoint_state(FLAGS.save_model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            tf.logging.info("Restore best model from " + checkpoint.model_checkpoint_path)
            model.saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            tf.logging.fatal("Checkpoint is not found, please check the best model save path is correct.")
            sys.exit(-1)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for batch in xrange(num_batches):
                if coord.should_stop():
                    break

                sep, mag_lengths = sess.run([model._sep, model._lengths])
                for i in xrange(FLAGS.batch_size):
                    filename = tfrecords_list[FLAGS.batch_size*batch+i]
                    (_, name) = os.path.split(filename)
                    (uttid, _) = os.path.splitext(name)
                    noisy_file = os.path.join(FLAGS.noisy_dir, uttid + '.wav')
                    enhan_sig, rate = reconstruct(np.squeeze(sep[i,:mag_lengths[i],:]), noisy_file)
                    savepath = os.path.join(FLAGS.rec_dir, uttid + '.wav')
                    wav.write(savepath, rate, enhan_sig)

                if (batch+1) % 100 == 0:
                    tf.logging.info("Number of batch processed: %d." % (batch+1))

        except Exception, e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)
        sess.close()

def main(_):
    if not os.path.exists(FLAGS.save_model_dir):
        tf.logging.fatal("The best model path is not exist, please check.")
        sys.exit(-1)

    if not os.path.exists(FLAGS.noisy_dir):
        tf.logging.fatal("The mixture speech path is not exist, please check. Use the phase of the mixture to reconstruct the separated speech.")
        sys.exit(-1)

    if not os.path.exists(FLAGS.rec_dir):
        os.makedirs(FLAGS.rec_dir)

    decode()

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
        '--inputs_cmvn',
        type=str,
        default='tfrecords/tr_cmvn.npz',
        help="The global cmvn to normalize the inputs."
    )
    parser.add_argument(
        '--noisy_dir',
        type=str,
        default='min/tt/mixed',
        help="The directory where the mixture speech is."
    )
    parser.add_argument(
        '--data_type',
        type=str,
        default='tt',
        help="The data type to decode (default is tt, it's the folder name where the mixture speech is saved)."
    )
    parser.add_argument(
        '--rec_dir',
        type=str,
        default='data/wav/rec/model_name',
        help="The directory where the separated speech is saved."
    )
    parser.add_argument(
        '--with_labels',
        type=int,
        default=1,
        help='Whether the clean labels are included in the tfrecords.'
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
        '--model_type',
        type=str,
        default='BLSTM',
        help="RNN model type."
    )
    parser.add_argument(
        '--mask_type',
        type=str,
        default='relu',
        help="Mask avtivation funciton, now only support sigmoid or relu"
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
        '--batch_size',
        type=int,
        default=16,
        help="Minibatch size."
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
        default=0.5,
        help="Keep probability for training with a dropout (default: 1-dropout_rate)."
    )
    parser.add_argument(
        '--FFT_LEN',
        type=int,
        default=256,
        help="The length of FFT."
    )
    parser.add_argument(
        '--FRAME_SHIFT',
        type=int,
        default=64,
        help="The frame shift."
    )
    FLAGS, unparsed = parser.parse_known_args()
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__dict__)
    sys.stdout.flush()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
