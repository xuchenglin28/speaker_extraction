# Target Speaker Extraction and Verification for Multi-talker Speech

The codes here are speaker extraction, where only target speaker's voice will be extracted given this target speaker's characteristics. In paper 2), we use a small network to jointly learn target speaker's characteristics from a different utterance of target speaker. You also can replace the network by using i-vector, or x-vector network.

If you are interested in speech separation to get all the speaker's voices in the mixture, please move to https://github.com/xuchenglin28/speech_separation

## Papers

Please cite:

 1) Chenglin Xu, Wei Rao, Xiong Xiao, Eng Siong Chng and Haizhou Li, "SINGLE CHANNEL SPEECH SEPARATION WITH CONSTRAINED UTTERANCE LEVEL PERMUTATION INVARIANT TRAINING USING GRID LSTM", in Proc. of ICASSP 2018, pp 6-10.
 2) Chenglin Xu, Wei Rao, Eng Siong Chng, and Haizhou Li, "Optimization of Speaker Extraction Neural Network with Magnitude and Temporal Spectrum Approximation Loss", in Proc. of ICASSP 2019, pp 6990-6994.
 3) Wei Rao, Chenglin Xu, Eng Siong Chng, and Haizhou Li, "Target Speaker Extraction for Overlapped Multi-Talker Speaker Verification", in Proc. of Interspeech 2019.

## Data Generation:

If you are using wsj0 to simulate data as in the paper 2) and 3), please read the code in run_data_generation.sh for detials, and change the path accordingly.

The list of files and SNRs for {training, development and test sets} are in simulation/mix_2_spk_{tr,cv,tt}\_extr.txt. In the files, the first column is the utterance of the target speaker to generate mixture and also used as target clean to supervise the network learning. The seconde column is the interference speaker to generate the mixture. The third column is the taget speaker's another utterane to obtain speaker's characteristics.

After run the .sh script, there will be 3 folders {mix, aux, s1} for the three sets {tr, cv, tt}. The mix folder is the mixture speech, aux folder is the utterances to obtain speaker's characteristics, and s1 is the folder of target clean. In all three folders, the names are cosistent for each example. 

## Speaker Extraction

This part includes feature extraction, model training, run-time inference. Please read the run.sh code for detail and revise accordingly.

noisy_dir: the folder where your simulated data is. For example, "data/wsj0_2mix_extr/wav8k/max", under this path, there will be three folder for training, development and test sets (tr, cv, tt). In each set, there will be three folder with names of (mix, aux, s1) as described in Data Genration part.

(The folder name for training and development set has been hard code. If you want to use differnt forlder name, please change parameters in read_list() function in train.py)

After given the path to the noisy_dir, you can just run the code to extract feature, train model, and do run-time inference.

   run.sh

## Speaker Verification: 

Here we only provide the key files for the paper 3) on speaker verification. Please read the paper for details.

verification/keys: key files of simulated trials for multi-talker speaker verification system.

## Environments:

python: 2.7

Tensorflow: 1.12 (some API are older version, but compatiable by 1.12)

## Contact

e-mail: xuchenglin28@gmail.com
