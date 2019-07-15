#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017   Chenglin Xu (NTU, Singapore)
# Updated by Chenglin, Dec 2018

"""
1. Extract features (magnitude, log magnitude)
2. Converts to TFRecords format
3. Calculate global CMVN (same as kaldi).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing
import os,sys
import numpy as np
import tensorflow as tf

from utils.audioread import audioread
from utils.sigproc import framesig,magspec
from utils.normhamming import normhamming
import time

def make_sequence(feats, feats_aux):
    """
    Return a sequence for given feats and corresponding labels (optional for test)
    Args:
        feats: input feature vectors (i.e. magnitude of mixture speech)
        feats_aux: inputs to auxilary network to learn target speaker representation
        labels1: reference labels for target sepaker
    Returns:
        A tf.train.SequenceExample
    """

    inputs = [tf.train.Feature(float_list=tf.train.FloatList(value=feat)) for feat in feats]
    inputs_aux = [tf.train.Feature(float_list=tf.train.FloatList(value=feat_aux)) for feat_aux in feats_aux]
    feature_list = {
        'inputs': tf.train.FeatureList(feature=inputs),
        'inputs_aux': tf.train.FeatureList(feature=inputs_aux)
    }

    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)

def cal_phase_mag(filename):
    '''
    extract phase and feats for one utterance
    '''
    
    rate, sig, _ = audioread(filename)
    frames = framesig(sig, FLAGS.FFT_LEN, FLAGS.FRAME_SHIFT, lambda x: normhamming(x), True)
    phase, feats = magspec(frames, FLAGS.FFT_LEN)

    return phase, feats

def extract_mag_feats(item):
    tokens = item.strip().split()

    (_, name) = os.path.split(tokens[0])
    (uttid, _) = os.path.splitext(name)  #mixed or noisy utterance
    # extract feats for mixture
    phase_mix, feats = cal_phase_mag(tokens[0])

    (_, name_aux) = os.path.split(tokens[1])
    (uttid_aux, _) = os.path.splitext(name_aux)
    tokens_aux = uttid_aux.split('_')
    # extract auxiliary feats for auxiliary network
    phase_aux, feats_aux = cal_phase_mag(tokens[1])

    # tfrecords to save the sequency consisting of feats and labels (optional for test)
    tfrecords_name = os.path.join(FLAGS.output_dir, FLAGS.data_type, uttid+".tfrecords")
    writer = tf.python_io.TFRecordWriter(tfrecords_name)
    # write feats and labels into tfrecords
    writer.write(make_sequence(feats, feats_aux).SerializeToString())

def main(unused_argv):
    print('Extract starts ...')
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))

    if not os.path.exists(os.path.join(FLAGS.output_dir, FLAGS.data_type)):
        os.makedirs(os.path.join(FLAGS.output_dir, FLAGS.data_type))

    lists = open(FLAGS.list_path).readlines()

    pool = multiprocessing.Pool(FLAGS.num_threads)
    workers = []
    for item in lists:
        workers.append(pool.apply_async(extract_mag_feats(item)))
    pool.close()
    pool.join()

    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
    print('Extract ends.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_type',
        type=str,
        default='tr',
        help='tr, cv, tt.')
    parser.add_argument(
        '--list_path',
        type=str,
        default='lists/rm4_tr.lst',
        help='List of the paired mix, aux, clean data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/tfrecords',
        help='Directory to save the features into tfrecords format'
    )
    parser.add_argument(
        '--FFT_LEN',
        type=int,
        default=512,
        help='The length of fft window.'
    )
    parser.add_argument(
        '--FRAME_SHIFT',
        type=int,
        default=256,
        help='The shift of samples when calculating fft.'
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=10,
        help='The number of threads to convert tfrecords files.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
