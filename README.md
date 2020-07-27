# About this repo
This repo contains the code and featurized data for paper 'Extracting Velocity-Based User-Tracking Features to Predict LearningGains in a Virtual Reality Training Application'.

# Files in repo
Two python scripts that are used to preprocessing the original data are presented in `data` folder. Since the data needs to be kept private, we are unable to reveal the original motion data. However, the featurized data will be provided.

Four python scripts are included in `classification` folder:
cvxfeature.py - a tiny libray used to extracting cvx based features from preprocessed data.
featurized.py - featurized the data given the hyper parameters, the data is a dictionary where the key is the hyperparameter ( segment_length, shift_size ,feature_para ), where the values is corresponding featurized data under this hyper parameter setting. 
find_segment.py - given the best hyper parameter configuration, find the motion segments from original data that the classifier believes most likely HL/LL gain users. 
tune_hyper.py - the main script for finding best hyper parameters using cross validation. 

All featurized data should be put into `classification/fdata` folder, since the size of data files are very large, I will provide external link for downloading it. 

# How to run the experiment
You will need `pyLeon` library to run this project. You can download this library from my homepage. 
The `tune_hyper.py` takes 3 args at least, the first one is the featurized data file, the second one is the normalization_flag (No normalization used in our case), while the third one is the split seed (0 used in our case), this controls how we divide the train/test users. 
For example, simply run `python tune_hyper.py fdata/cvx9.pkl 0 0` will find the best hyperparameter for CVX feature when we only consider linear velocity of HMD and controller.
