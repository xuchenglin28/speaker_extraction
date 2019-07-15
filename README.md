Target Speaker Extraction and Verification for Multi-talker Speech

Please cite:

 1) Chenglin Xu, Wei Rao, Xiong Xiao, Eng Siong Chng and Haizhou Li, "SINGLE CHANNEL SPEECH SEPARATION WITH CONSTRAINED UTTERANCE LEVEL PERMUTATION INVARIANT TRAINING USING GRID LSTM", in Proc. of ICASSP 2018, pp 6-10.
 2) Chenglin Xu, Wei Rao, Eng Siong Chng, and Haizhou Li, "Optimization of Speaker Extraction Neural Network with Magnitude and Temporal Spectrum Approximation Loss", in Proc. of ICASSP 2019, pp 6990-6994.
 3) Wei Rao, Chenglin Xu, Eng Siong Chng, and Haizhou Li, "Target Speaker Extraction for Overlapped Multi-Talker Speaker Verification", in Proc. of Interspeech 2019.

Folder:

1. simulation: run_data_generation.sh
   code to generate a 2-speaker mixed database for target speaker extraction.
2. feature extraction, model training, run-time inference:
   run.sh
3. verification: 
   key files of simulated trials for multi-talker speaker verification system.

Environments:
python: 2.7
Tensorflow: 1.12 (some API are older version, but compatiable by 1.12)
