#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Chenglin Xu (NTU,Singapore)


"""Utility functions for computing dynamic features."""

import tensorflow as tf
#import numpy as np

def comp_dynamic_feature(inputs, DELTAWINDOW, Batch_size, lengths):
	#sess = tf.Session()
	#N_bat = tf.shape(inputs)
	#print sess.run(N_bat[0])
	#print sess.run(tf.size(inputs))
	#inputs = tf.contrib.util.constant_value(inputs)
	#print(inputs)
	#outputs = tf.zeros(N_bat, dtype=tf.float32)
	#for i in range(N_bat[0]):
	#	tmp = comp_delta(inputs[i,:,:], DELTAWINDOW)
	#	outputs[i,:,:] = tmp;
	#return tf.convert_to_tensor(outputs)
	#N_bat, N_vec, N_cep = np.shape(tf.contrib.util.constant_value(inputs))
	#print Batch_size
	#outputs = tf.Variable(inputs, trainable=False)
	outputs = []
	for i in range(Batch_size):
		tmp = comp_delta(inputs[i,:lengths[i],:], DELTAWINDOW, lengths[i])
		#with tf.Session() as sess:
		#	print(sess.run(tmp))
		#outputs[i,:,:] = tmp;
		#tf.assign(outputs[i,:,:],tf.convert_to_tensor(tmp));
		tmp1 = tf.pad(tmp, [[0,tf.reduce_max(lengths)-lengths[i]],[0,0]], "CONSTANT")
		outputs.append(tmp1)
	#print(outputs)
	return tf.convert_to_tensor(outputs)


#def comp_delta(static_coef, DELTAWINDOW):
#	N_vec, N_cep = np.shape(static_coef)
#	static_coef = np.repeat(static_coef, np.r_[DELTAWINDOW+1,[1]*(N_vec-2),DELTAWINDOW+1], 0)
#
#	delta_coef = 0.0
#	i = DELTAWINDOW+1
#	denom = np.sum(np.power(range(1,DELTAWINDOW+1),2))*2.0
#	for j in range(1,DELTAWINDOW+1):
#		delta_coef += j/denom*(static_coef[i+j-1:i+j+N_vec-1,:]-static_coef[i-j-1:i-j+N_vec-1,:])
#	return delta_coef

def comp_delta(feat, N, length):
	"""Compute delta features from a feature vector sequence.
	Args:
		feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
		N: For each frame, calculate delta features based on preceding and following N frames.
	Returns:
		A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    	"""
	#NUMFRAMES= tf.shape(feat)
	#with tf.Session() as sess:
	#	sess.run(NUMFRAMES)
	#print(tf.size(feat))
	#print feat
	#feat = np.concatenate(([feat[0] for i in range(N)], feat, [feat[-1] for i in range(N)]))
	feat = tf.concat([[feat[0] for i in range(N)], feat, [feat[-1] for i in range(N)]], 0)
	#with tf.Session() as sess:
	#	sess.run(tf.initialize_all_variables())
	#	print(sess.run(tf.shape(feat)))
	#	print(sess.run(length))
	denom = sum([2*i*i for i in range(1,N+1)])
	#dfeat = []
	#for j in tf.range(length):
	#	#dfeat.append(np.sum([n*feat[N+j+n] for n in range(-1*N,N+1)], axis=0)/denom)
	#	dfeat.append(tf.reduce_sum([n*feat[N+j+n] for n in range(-1*N,N+1)], axis=0)/denom)
	dfeat = tf.reduce_sum([j*(feat[N+1+j-1:N+1+j+length-1,:]-feat[N+1-j-1:N+1-j+length-1,:]) for j in range(1,N+1)], axis=0)/denom
	return tf.convert_to_tensor(dfeat)

#def main():
#	#delta_coef = comp_dynamic_feature(np.array([[[1.0,2],[3,4],[5,6],[7,8]],[[1.0,2],[3,4],[5,6],[7,8]]]), 2, 2)
#	#print delta_coef
#	sess = tf.Session()
#	delta_coef = comp_dynamic_feature(tf.convert_to_tensor([[[1.0,2],[3,4],[5,6],[7,8],[0,0]],[[1.0,2],[3,4],[5,6],[7,8],[9,10]]], dtype=tf.float32), 2, 2, [4,5])
#	print(sess.run(delta_coef))
#	sess.close()
#
#if __name__ == '__main__':
#	main()

#def main():
#
#	#delta_coef = comp_delta([[1.0,2],[3,4],[5,6],[7,8]], 2)
#	delta_coef = comp_dynamic_feature(np.array([[[1.0,2],[3,4],[5,6],[7,8]],[[1.0,2],[3,4],[5,6],[7,8]]]), 2)
#	print delta_coef
#
#if __name__ == '__main__':
#	main()
