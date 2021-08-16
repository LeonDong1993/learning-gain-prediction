# About
This repo contains the code and featurized data for paper 'Extracting Velocity-Based User-Tracking Features to Predict LearningGains in a Virtual Reality Training Application' published in 2020 IEEE International Symposium on Mixed and Augmented Reality (ISMAR).

# Details
Two python scripts that are used to preprocessing the original data are presented in `data` folder. Since the data needs to be kept private, we are unable to reveal the original motion data. However, the featurized data will be provided.

Four python scripts are included in `classification` folder:
- cvxfeature.py - a tiny libray used to extracting cvx based features from preprocessed data.
- featurized.py - featurized the data given the hyper parameters, the data is a dictionary where the key is the hyperparameter ( segment_length, shift_size ,feature_para ), where the values is corresponding featurized data under this hyper parameter setting.
- find\_segment.py - given the best hyper parameter configuration, find the motion segments from original data that the classifier believes most likely HL/LL gain users.
- tune\_hyper.py - the main script for finding best hyper parameters using cross validation.

All featurized data should be put into `classification/fdata` folder, since the size of data files are very large, it can be download at https://drive.google.com/file/d/1MkxTrAqCN_wBiqRToj1XobRKC5WGcFNF/view?usp=sharing. The file contains only two files `cvx21.pkl` and `pca21.pkl`.
- `cvx21.pkl` is the featurized data using CVX feature extraction method over all attributes, i.e, linear and angular velocities of  HMD and both hands.
- `pca21.pkl` is the featurized data using PCA feature extraction method over all attributes, i.e, linear and angular velocities of  HMD and both hands.

You can create featurized data using PCA/CVX feature extraction method over a subset of attributes by using the `create_partial_data.py` script provided in `classification/fdata` directory.
The usage of the script is `python create_partial_data.py source_file output_file drop_attr_ids`. The drop_attr_ids is predefined as
- 0-linear velocities of HMD
- 1-angular velocities of HMD
- 2-linear velocities of left hand
- 3-angular velocities of left hand
- 4-linear velocities of right hand
- 5-angular velocities of right hand

For example, `python create_partial_data.py cvx21.pkl cvx13.pkl 3,5` will create a featurized data using  CVX feature extraction method over linear and angular velocities of  HMD and only linear velocities of both hands.

# How to Run
You will need to source the `set_env.profile` file first before run the project. Otherwise, it will throw out library not found error.

The `tune_hyper.py` takes 3 args at least, the first one is the featurized data file, the second one is the normalization_flag (No normalization used in our case), while the third one is the split seed (0 used in our case), this controls how we divide the train/test users.  For example, simply run `python tune_hyper.py fdata/cvx9.pkl 0 0` will find the best hyperparameter for CVX feature when we only consider linear velocity of HMD and controller.
