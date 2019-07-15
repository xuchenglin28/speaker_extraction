#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017   Chenglin Xu (NTU, Singapore)
# Updated by Chenglin, Dec 2018, Jul 2019

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

def make_sequence(feats, feats_aux, labels=None):
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
    if labels is not None:
        targets = [tf.train.Feature(float_list=tf.train.FloatList(value=label)) for label in labels]
        feature_list = {
            'inputs': tf.train.FeatureList(feature=inputs),
            'inputs_aux': tf.train.FeatureList(feature=inputs_aux),
            'labels': tf.train.FeatureList(feature=targets)
        }
    else:
        feature_list = {
            'inputs': tf.train.FeatureList(feature=inputs),
            'inputs_aux': tf.train.FeatureList(feature=inputs_aux)
        }

    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)

def cal_phase_mag(filename, dur=None):
    '''
    extract phase and feats for one utterance
    '''
    
    rate, sig, _ = audioread(filename)
    if dur != 0:
        sig = sig[:rate*dur]
    frames = framesig(sig, FLAGS.FFT_LEN, FLAGS.FRAME_SHIFT, lambda x: normhamming(x), True)
    phase, feats = magspec(frames, FLAGS.FFT_LEN)

    return phase, feats

def cal_intermedia_mean_var(feats):
    mean_feats = np.sum(feats, 0)
    var_feats = np.sum(np.square(feats), 0)                                                      
    return str(np.shape(feats)[0])+'+'+' '.join(str(mean_feat) for mean_feat in mean_feats)+'+'+' '.join(str(var_feat) for var_feat in var_feats)

def extract_mag_feats(item, mean_var_dict, mean_var_dict_aux):
    tokens = item.strip().split()

    (_, name) = os.path.split(tokens[0])
    (uttid, _) = os.path.splitext(name)
    # extract feats for mixture
    phase_mix, feats = cal_phase_mag(tokens[0], dur=FLAGS.dur)
    mean_var_dict[uttid] = cal_intermedia_mean_var(feats)

    (_, name_aux) = os.path.split(tokens[1])
    (uttid_aux, _) = os.path.splitext(name_aux)
    tokens_aux = uttid_aux.split('-')
    # extract auxiliary feats for auxiliary network
    phase_aux, feats_aux = cal_phase_mag(tokens[1], dur=FLAGS.dur)
    # calculate intermediates for mean and variance for auxiliary inputs, save to kaldi vector format
    mean_var_dict_aux[uttid] = cal_intermedia_mean_var(feats_aux)

    # extract mag for clean as labels
    if FLAGS.with_labels:
        # extract feats for mixture
        phase_clean, labels = cal_phase_mag(tokens[2], dur=FLAGS.dur)

        if FLAGS.apply_psm:
            labels = labels * np.cos(phase_mix - phase_clean)
    else:
        labels = None
    
    # tfrecords to save the sequency consisting of feats and labels (optional for test)
    tfrecords_name = os.path.join(FLAGS.output_dir, FLAGS.data_type, uttid+".tfrecords")
    writer = tf.python_io.TFRecordWriter(tfrecords_name)
    # write feats and labels into tfrecords
    writer.write(make_sequence(feats, feats_aux, labels).SerializeToString())

    return mean_var_dict, mean_var_dict_aux

def cal_global_mean_std(filename, mean_var_dict):
    cmvn = np.zeros((2, int(FLAGS.FFT_LEN/2+1)), dtype=np.float32)
    frames = 0.0
    for line in mean_var_dict:
        tokens = line.strip().split('+')
        frames += float(tokens[0])
        utt_mean_tokens = tokens[1].strip().split()
        cmvn[0] += [np.float32(i) for i in utt_mean_tokens]
        utt_var_tokens = tokens[2].strip().split()
        cmvn[1] += [np.float32(i) for i in utt_var_tokens]

    mean = cmvn[0] / frames
    var = cmvn[1] / frames - mean ** 2
    var[var<=0] = 1.0e-20
    std = np.sqrt(var)

    print(mean)
    print(std)
    np.savez(filename, mean_inputs=mean, stddev_inputs=std)

def main(unused_argv):
    print('Extract starts ...')
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))

    if not os.path.exists(os.path.join(FLAGS.output_dir, FLAGS.data_type)):
        os.makedirs(os.path.join(FLAGS.output_dir, FLAGS.data_type))

    lists = open(FLAGS.list_path).readlines()

    # check whether the cmvn file for training exist, remove if exist.
    if os.path.exists(FLAGS.inputs_cmvn):
        os.remove(FLAGS.inputs_cmvn)
    if os.path.exists(FLAGS.inputs_cmvn.replace('cmvn', 'cmvn_aux')):
        os.remove(FLAGS.inputs_cmvn.replace('cmvn', 'cmvn_aux'))

    mean_var_dict = multiprocessing.Manager().dict()
    mean_var_dict_aux = multiprocessing.Manager().dict()
    pool = multiprocessing.Pool(FLAGS.num_threads)
    workers = []
    for item in lists:
        workers.append(pool.apply_async(extract_mag_feats(item, mean_var_dict, mean_var_dict_aux)))
    pool.close()
    pool.join()

    # convert the utterance level intermediates for mean and var to global mean and std, then save
    cal_global_mean_std(FLAGS.inputs_cmvn, mean_var_dict.values())
    cal_global_mean_std(FLAGS.inputs_cmvn.replace('cmvn', 'cmvn_aux'), mean_var_dict_aux.values())
    
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
    print('Extract ends.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--with_labels',
        type=int,
        default=1,
        help='Whether extract features for the targets as labels, default to prepare labels.')
    parser.add_argument(
        '--data_type',
        type=str,
        default='tr',
        help='tr, cv, tt.')
    parser.add_argument(
        '--apply_psm',
        type=int,
        default=1,
        help='Whether use phase sensitive mask.')
    parser.add_argument(
        '--inputs_cmvn',
        type=str,
        default='data/inputs_utts.cmvn',
        help='Path to save CMVN for the inputs'
    )
    parser.add_argument(
        '--list_path',
        type=str,
        default='lists/tr_mix.lst',
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
    parser.add_argument(
        '--dur',
        type=int,
        default=0,
        help='Duration of each file, cut to fixed length wav for mix, aux, clean'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
