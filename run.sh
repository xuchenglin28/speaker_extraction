#! /bin/bash

# Copyright 2017
# Author: Chenglin Xu (NTU, Singapore)
# Email: xuchenglin28@gmail.com
# Updated by Chenglin, Dec 2018, Jul 2019
#Please cite:
#   Chenglin Xu, Wei Rao, Xiong Xiao, Eng Siong Chng and Haizhou Li, "SINGLE CHANNEL SPEECH SEPARATION WITH CONSTRAINED UTTERANCE LEVEL PERMUTATION INVARIANT TRAINING USING GRID LSTM", in Proc. of ICASSP 2018, pp 6-10.
#  Chenglin Xu, Wei Rao, Eng Siong Chng, and Haizhou Li, "Optimization of Speaker Extraction Neural Network with Magnitude and Temporal Spectrum Approximation Loss", in ICASSP 2019.
#  Wei Rao, Chenglin Xu, Eng Siong Chng, and Haizhou Li, "Target Speaker Extraction for Overlapped Multi-Talker Speaker Verification", in Interspeech 2019.

step=1
gpu_id=$1 #'0', '1'

# Paths for reading wav and saving features
noisy_dir=data/wsj0_2mix_extr/wav8k/max
rec_dir=data/rec #save extraced speech
tfrecords_dir=data/tfrecords/mag_mix

# Configure for feature extraction
FFT_LEN=256
FRAME_SHIFT=128
with_labels=1 # set to 0 when run-time inference, no available labels
apply_psm=1
dur=4 # whether cut the utterances of training and development sets into segments. 0 should be set during run-time (NO CUT)

# Configure for network
rnn_num_layers=1
input_size=129
output_size=129
rnn_size=512
model_type=BLSTM
mask_type=relu

aux_hidden_size=256
aux_output_size=30

# Configure for objective function
mag_factor=1.0
del_factor=4.5
acc_factor=10.0
dynamic_win=2
power_num=2 # mean square error (=2) or mean absolute error (=1)

# Configure for training
TF_CPP_MIN_LOG_LEVEL=1
tr_batch_size=2 # [change to situable size according to your GPU memory]
tt_batch_size=1
keep_prob=0.5
learning_rate=0.0005
lr_reduction_factor=0.7
min_epochs=50
max_epochs=200
num_threads=10

# Path to save model and reconstructed wav
name=Mag_Mix_Cat_${model_type}_${rnn_num_layers}_${rnn_size}_a${aux_hidden_size}_${aux_output_size}_${mask_type}_del${del_factor}_acc${acc_factor}_win${dynamic_win}_L${power_num}
save_model_dir=exp/$name/

lists_dir=./tmp/mag_lists_mix
mkdir -p $lists_dir

echo "FRAME_SHIFT=$FRAME_SHIFT" > config.py

#################################################
# generate data before run this script. 
# data simulation script, run_data_generation.sh
#################################################

if [ $step -le 1 ]; then
    echo "Prepare data"
    for x in tr cv tt; do
        ls $noisy_dir/$x/mix/*.wav | awk '{a=$1; gsub("/mix/", "/aux/", $1); b=$1; gsub("/aux/", "/s1/", $1); printf("%s %s %s\n", a, b, $1)}' > ${lists_dir}/${x}_mix.lst &
    done
    wait

    for x in tr cv; do
        python extract_feats.py --data_type=$x --inputs_cmvn=$tfrecords_dir/${x}_cmvn.npz --with_labels=$with_labels --apply_psm=$apply_psm --list_path=${lists_dir}/${x}_mix.lst --output_dir=$tfrecords_dir --FFT_LEN=$FFT_LEN --FRAME_SHIFT=$FRAME_SHIFT --num_threads=$num_threads --dur=$dur &
    done
    wait
    echo "Prepare data done."
fi

# Training
if [ $step -le 2 ]; then
    echo "Model training starts."
    # sort the tfrecord files by size in order to group the utterances with similar length into a minibatch
    for x in tr cv; do
        ls -Sr $tfrecords_dir/${x}/*.tfrecords > $lists_dir/${x}.lst &
    done
    wait
    shuffle=0
    command="python train.py --lists_dir=$lists_dir --save_model_dir=$save_model_dir --with_labels=$with_labels --rnn_num_layers=$rnn_num_layers --rnn_size=$rnn_size \
        --input_size=$input_size --output_size=$output_size --mask_type=$mask_type --aux_hidden_size=$aux_hidden_size --aux_output_size=$aux_output_size \
        --batch_size=$tr_batch_size --lr_reduction_factor=$lr_reduction_factor --learning_rate=$learning_rate --keep_prob=$keep_prob \
        --inputs_cmvn=$tfrecords_dir/tr_cmvn.npz --min_epochs=$min_epochs --max_epochs=$max_epochs --num_threads=$num_threads \
        --del_factor=$del_factor --acc_factor=$acc_factor --dynamic_win=$dynamic_win --mag_factor=$mag_factor --power_num=$power_num --shuffle=$shuffle "

    echo $command
    CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $command
    echo "Model training ends."
fi

######################################
# Prepare data and decode for tt
######################################
if [ $step -le 3 ]; then
    echo "Prepare data"
    lists_name=tt
    mkdir -p $tfrecords_dir/${lists_name}
    python extract_feats_test.py --data_type=${lists_name} --list_path=${lists_dir}/${lists_name}_mix.lst --output_dir=$tfrecords_dir --FFT_LEN=$FFT_LEN --FRAME_SHIFT=$FRAME_SHIFT --num_threads=$num_threads
    echo "Prepare data done."
fi

# Decoding
if [ $step -le 4 ]; then

    echo "Start Decoding."
    lists_name=tt
    find $tfrecords_dir/$lists_name/ -iname "*.tfrecords" > $lists_dir/${lists_name}.lst
    num_threads=1
    with_labels=0
    inputs_cmvn=$tfrecords_dir/tr_cmvn.npz
    mkdir -p $rec_dir
    
    decode_cmd="python -u decode.py --lists_dir=$lists_dir --data_type=${lists_name} --noisy_dir=$noisy_dir/$lists_name/mix --save_model_dir=$save_model_dir --with_labels=$with_labels \
        --rec_dir=$rec_dir/$lists_name/$name --rnn_num_layers=$rnn_num_layers --rnn_size=$rnn_size --input_size=$input_size --output_size=$output_size --aux_hidden_size=$aux_hidden_size \
        --aux_output_size=$aux_output_size --mask_type=$mask_type --keep_prob=$keep_prob --batch_size=$tt_batch_size --inputs_cmvn=$inputs_cmvn \
        --num_threads=$num_threads --FFT_LEN=$FFT_LEN --FRAME_SHIFT=$FRAME_SHIFT --power_num=$power_num "
    
    echo $decode_cmd
    CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $decode_cmd
    
    echo "Decoding ends"
fi
