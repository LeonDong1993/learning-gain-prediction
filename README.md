# About
This repo contains the code and **featurized** data for the paper `Extracting Velocity-Based User-Tracking Features to Predict LearningGains in a Virtual Reality Training Application` published in 2020 IEEE International Symposium on Mixed and Augmented Reality (ISMAR 2020).

In this work, we leverage machine learning methods to improve the ability of evaluating learning outcomes in VR training scenarios. Specifically, we investigate the feasibility of using a machine learning approach to classify users of a VR training system into
groups of low-learning (LL) and high-learning (HL) gains, based on their head-mounted display (HMD) and controller tracking data only.

# Getting Started
## Setup
Two python scripts are used to preprocessing the original data and the results are stored in `data` folder. Note that the original user data needs to be kept private, but the featurized data will be provided.

Four python scripts are included in the `classification` folder:
- `cvxfeature.py` - a tiny libray used to extracting CVX based features from preprocessed data.
- `featurize.py` - featurize the data given the hyper parameters, the data is a dictionary where the key is the hyperparameter `(segment_length, shift_size, feature_para)`, and the values is the corresponding featurized data under this hyperparameter setting.
- `find_segment.py` - given the best hyper parameter configuration, find the motion segments from original data that the classifier believes most likely HL/LL gain users.
- `tune_hyper.py` - the **main entry script** for training and model and finding best hyper parameters through cross validation.

You need to source the `set_env.profile` first before run the project, otherwise you may encounter library not found error.
> You might also need to add the project directory to the python path (e.g. the Linux command export PYTHONPATH=$(pwd):$PYTHONPATH).

The `tune_hyper.py` takes at least three arguments:
- the first one is the featurized data file
- the second one is the normalization_flag (No normalization used in our case)
- the third one is the split seed (0 used in our case), this controls how we divide the train/test users.


## Data
All featurized data should be put into `classification/fdata` folder, and they can be downloaded from [HERE][data_url]. And it contains two files `cvx21.pkl` and `pca21.pkl`.
- `cvx21.pkl` is the featurized data using CVX feature extraction method over all attributes, i.e, linear and angular velocities of  HMD and both hands.
- `pca21.pkl` is the featurized data using PCA feature extraction method over all attributes, i.e, linear and angular velocities of  HMD and both hands.

You can create featurized data using PCA/CVX feature extraction method over a subset of attributes by using the `create_partial_data.py` script provided in `classification/fdata` directory.

The usage of the script is `python create_partial_data.py source_file output_file drop_attr_ids`. The `drop_attr_ids` is predefined as
- 0-linear velocities of HMD
- 1-angular velocities of HMD
- 2-linear velocities of left hand
- 3-angular velocities of left hand
- 4-linear velocities of right hand
- 5-angular velocities of right hand

For example, `python create_partial_data.py cvx21.pkl linear_velo.pkl 1,3,5` will create a featurized data using  CVX feature extraction method that have only the featurized data over linear velocities of HMD and both hands.


# Results
We evaluated the performance of our machine learning model using different tracking feature combinations as well as different featurization methods.

In short, we misclassify at most 1 users in the test set, and our model achieved high confidence in its prediction and able to identifying all high learning gain users in the test set. 

![experiment results on test set][res_fig]




# Citation
Please cite our work if you find it is helpful for your research!

```
@INPROCEEDINGS{9284660,
  author={Moore, Alec G. and McMahan, Ryan P. and Dong, Hailiang and Ruozzi, Nicholas},
  booktitle={2020 IEEE International Symposium on Mixed and Augmented Reality (ISMAR)}, 
  title={Extracting Velocity-Based User-Tracking Features to Predict Learning Gains in a Virtual Reality Training Application}, 
  year={2020},
  volume={},
  number={},
  pages={694-703},
  keywords={Training;Solid modeling;Resists;Machine learning;Feature extraction;Task analysis;Testing;Feature extraction;machine learning;virtual reality;Human-centered computing;Human computer interaction (HCI);Interaction paradigms;Virtual reality;Computing methodologies;Machine learning;Machine learning algorithms;Feature selection},
  doi={10.1109/ISMAR50242.2020.00099}}
```


# Contact
If you have any questions or need help regarding our work, you can email us and we are happy to discuss the work (the email addresses of each author are included in the paper). 

In case my school email being deactivated, you can email me using my personal email address `HailiangDong@hotmail.com`.


[data_url]:https://utdallas.box.com/s/0trds4nsf6p4wc9fwuze89oukmloyjav
[res_fig]:https://github.com/LeonDong1993/learning-gain-prediction/blob/master/figs/vr-res.png



