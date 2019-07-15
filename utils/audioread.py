#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017  Chenglin Xu
# Updated by Chenglin, Dec 2018, Jul 2019

"""
audioread function, same as matlab
"""

import scipy.io.wavfile as wav

def audioread(filename):
    (rate,sig) = wav.read(filename)
    if sig.dtype == 'int16':
        nb_bits = 16
    elif sig.dtype == 'int32':
        nb_bits = 32
    sig = sig / float(2 ** (nb_bits - 1))
    
    return rate, sig, nb_bits
