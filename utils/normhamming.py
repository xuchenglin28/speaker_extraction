#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017  Chenglin Xu

"""
normalized square root hamming periodic window
"""

import numpy
from scipy.signal import hamming
from config import *

#FRAME_SHIFT=64
def normhamming(fft_len):
    win = numpy.sqrt(hamming(fft_len, False))
    win = win/numpy.sqrt(numpy.sum(numpy.power(win[0:fft_len:FRAME_SHIFT],2)))
    return win
