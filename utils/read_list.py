#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017  Chenglin Xu (NTU, Singapore)

"""
    Build the BLSTM and Grid LSTM model for cuPIT monaural speech separation.
    Please cite: 
      Chenglin Xu, Wei Rao, Xiong Xiao, Eng Siong Chng and Haizhou Li, 
      "SINGLE CHANNEL SPEECH SEPARATION WITH CONSTRAINED UTTERANCE LEVEL PERMUTATION INVARIANT TRAINING USING GRID LSTM",
      in ICASSP 2018.
"""

import os, sys
import numpy as np
import tensorflow as tf

def read_list(lists_dir, name, batch_size):
    file_name = os.path.join(lists_dir, name + ".lst")
    if not os.path.exists(file_name):
        tf.logging.fatal("The file list %s doesn't exist", file_name)
        sys.exit(-1)
    lines = open(file_name, 'r').readlines()
    tfrecords_list = []
    for line in lines:
        utt_id = line.strip().split()[0]
        if not os.path.exists(utt_id):
            tf.logging.fatal("TFRecords file %s doesn't exist", utt_id)
            sys.exit(-1)
        tfrecords_list.append(utt_id)
    num_batches = int(np.ceil(len(tfrecords_list) / batch_size))
    
    return tfrecords_list, num_batches

