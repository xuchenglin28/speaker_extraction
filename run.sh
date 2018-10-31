#! /bin/bash

# Copyright 2018  Chenglin Xu (NTU, Singapore)
# 
# Speech extraction code for data simultation, model training and testing. (model related code will be released later)
#
# Please cite: 
#   1. Chenglin Xu, Wei Rao, Eng Siong Chng, and Haizhou Li, "Optimization of Speaker Extraction Neural Network with Magnitude and Temporal Spectrum Approximation Loss", submitted to ICASSP 2019.
#   2. Wei Rao, Chenglin Xu, Eng Siong Chng, and Haizhou Li, "Target Speaker Extraction for Overlapped Multi-Talker Speaker Verification", submitted to ICASSP 2019.

step=0


if [ $step -le 0 ]; then

    #1. Assume that WSJ0's wv1 sphere files is converted to wav files. The folder
    #   structure and file name are kept same under wsj0/, e.g.,
    #   'ORG_PATH/wsj0/si_tr_s/01t/01to030v.wv1' is converted to wav and
    #   stored in 'YOUR_PATH/wsj0/si_tr_s/01t/01to030v.wv1'.
    #   Relevant data ('si_tr_s', 'si_dt_05' and 'si_et_05') are under YOUR_PATH/wsj0/
    #2. Put 'voicebox' toolbox in current folder. (http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html)
    #3. Set your 'YOUR_PATH' and 'OUTPUT_PATH' properly, then run this script in Matlab.
    #   (The max lenght of the wav will be kept when generate the mixture. The sampling rate will be 8kHz.)

    echo "Start to simulate data."
    
    cd ./simulation
    wsj0root='/media/clx214/data/wsj/' #YOUR_PATH
    output_dir='/media/clx214/data/wsj0_2mix_extr_tmp/wav8k' #OUTPUT_PATH to save simulated data
    fs8k=8000
    min_max='max'

    for data_type in tr cv tt; do
        matlab -nodesktop -nosplash -r "addpath('/media/clx214/ssd2/Projects/tfextraction/simulation/voicebox'); simulate_2spk_mix('$data_type', '$wsj0root', '$output_dir', $fs8k, '$min_max'); exit;" &
    done
    wait

    cd ../
fi
