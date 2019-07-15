#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017   Xu Chenglin(NTU, Singapore)
# Updated by Chenglin, Dec 2018, Jul 2019

import tensorflow as tf

def paddedFIFO_batch(file_list, batch_size, input_size, output_size, cmvn=None, cmvn_aux=None, with_labels=1, num_enqueuing_threads=4, 
                         num_epochs=1, capacity=1000, shuffle=False):
    """
    Pad several utterances (batch_size) into a batch with the length equal to the longest by zero.
    Args:
        file_list: A list of TFRecord files (to speed up, the list is sorted according of the utt length).
        batch_size: The number of sequential examples in a batch.
        input_size: The dimension of input feature.
        output_size: The dimension of target reference.
        cmvn: The mean and standard deviation used to normalize the input feature.
        num_enqueuing_threads: The number of threads used to enqueue.
        num_epochs: The number of epochs to train the model.
        shuffle: Whether to shuffle the utts.
    Returns:
        inputs: The input mixture ([batch_size, time_steps, input_size]).
        inputs_norm: The inputs to main network, which are normalized mixture ([batch_size, time_steps, input_size]).
        inputs_norm_aux: The normalized inputs to auxiliary network([batch_size, time_steps, input_size]).
        labels: The target reference of speaker 1 ([batch_size, time_steps, input_size]).
        length: The length of each utt before padding [batch_size].
        length_aux: The length of each auxiliary utt before padding [batch_size].
    """
    
    file_queue = tf.train.string_input_producer(file_list, num_epochs=num_epochs, shuffle=shuffle)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    
    if with_labels:
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[input_size], dtype=tf.float32),
            'inputs_aux': tf.FixedLenSequenceFeature(shape=[input_size], dtype=tf.float32),
            'labels': tf.FixedLenSequenceFeature(shape=[output_size], dtype=tf.float32)
        }
    else:
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[input_size], dtype=tf.float32),
            'inputs_aux': tf.FixedLenSequenceFeature(shape=[input_size], dtype=tf.float32)
        }
    _, sequence = tf.parse_single_sequence_example(serialized_example, sequence_features=sequence_features)
    length = tf.shape(sequence['inputs'])[0]
    length_aux = tf.shape(sequence['inputs_aux'])[0]
    if cmvn is not None and cmvn_aux is not None:
        # global mean and std
        inputs_norm = (sequence['inputs'] - cmvn['mean_inputs']) / (cmvn['stddev_inputs'] + 0.000001)
        inputs_norm_aux = (sequence['inputs_aux'] - cmvn_aux['mean_inputs']) / (cmvn_aux['stddev_inputs'] + 0.000001)
    else:
        mean, var = tf.nn.moments(sequence['inputs'], axes=[0])
        inputs_norm = (sequence['inputs'] - mean) / (tf.sqrt(var) + 0.000001)

        mean_aux, var_aux = tf.nn.moments(sequence['inputs_aux'], axes=[0])
        inputs_norm_aux = (sequence['inputs_aux'] - mean_aux) / (tf.sqrt(var_aux) + 0.000001)

    if with_labels:
        # tf.io.PaddingFIFOQueue in r1.12
        queue = tf.PaddingFIFOQueue(capacity=capacity, dtypes=[tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.int32],
                    shapes=[(None, input_size), (None, input_size), (None, input_size), (None, output_size), (), ()])
        enqueue_ops = [queue.enqueue([sequence['inputs'], inputs_norm, inputs_norm_aux, sequence['labels'], length, length_aux])] * num_enqueuing_threads
    else:
        # tf.io.PaddingFIFOQueue in r1.12
        queue = tf.PaddingFIFOQueue(capacity=capacity, dtypes=[tf.float32, tf.float32, tf.float32, tf.int32, tf.int32],
                    shapes=[(None, input_size), (None, input_size), (None, input_size), (), ()])
        enqueue_ops = [queue.enqueue([sequence['inputs'], inputs_norm, inputs_norm_aux, length, length_aux])] * num_enqueuing_threads

    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))
    return queue.dequeue_many(batch_size)
