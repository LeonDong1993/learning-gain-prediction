# coding: utf-8
import os, sys, random
import numpy as np
from multiprocessing import Pool
from collections import defaultdict
from pdb import set_trace
from copy import deepcopy
from functools import partial
from tqdm import tqdm

from pyLeon import utils
from pyLeon.misc import Logger

from sklearn.svm import SVC
from sklearn.metrics import classification_report

sys.path.insert(0,'../data/')
import mlib_data
import gendata

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

MAX_CORE = None
EPS = 1e-5

def split_users(users,ratio = 0.8, seed = -1):
	inds = defaultdict(list)
	for uid,score in users:
		inds[score].append(uid)

	train = []
	test = []
	for k,v in inds.items():
		N = len(v)
		if seed > 0:
			print("Randomization conducted!")
			random.Random(seed).shuffle(v)
		splitter = round(ratio*N)
		train += v[0:splitter]
		test += v[splitter:]
		print('Class {} has {} users, {} goes to train'.format(k,N,splitter))
		print('Test users ID is {}'.format(v[splitter:]))
	return train,test

def split_into_groups(array_data, groups):
	X,Y,T,S = array_data
	ret = []
	for gp in groups:
		ind_list = [np.where(T == uid)[0] for uid in gp]
		inds = np.concatenate(ind_list)
		subX = X[inds,:]
		subY = Y[inds]
		subT = T[inds]
		subS = S[inds]
		ret.append( (subX,subY,subT,subS) )
	return ret

def run_train_valid(args):
	params, train, valid = args
	svm_C, svm_gamma = params
	svc = SVC(class_weight = CLASS_WEIGHT, C = svm_C, gamma = svm_gamma).fit(train[0],train[1])
	pred = svc.predict(valid[0])
	result = classification_report(valid[1],pred, output_dict = True)
	result['seq_perf'] = sequence_accuracy(svc,valid)
	del result['weighted avg']
	del result['macro avg']
	return result

def std_cross_valid(args):
	params, train = args
	# svm_C,svm_gamma = params
	X,Y,T,S = train
	num_fold = STD_CV_FOLD
	# calculate the inds for each fold
	fold_inds = defaultdict(list)
	classes = np.unique(Y)
	for c in classes:
		inds = np.where(Y==c)[0]
		users = np.unique(T[inds])
		N_user = users.size
		max_fold_num = N_user % num_fold
		max_fold_size = int(N_user/num_fold)+1
		# split users into num_fold groups
		st = 0
		for i in range(num_fold):
			if i < max_fold_num:
				ed = st + max_fold_size
			else:
				ed = st + max_fold_size -1
			group = users[st:ed]
			st = ed
			for uid in group:
				fold_inds[i].append( np.where(T == uid)[0] )

	for k in fold_inds:
		fold_inds[k] = np.concatenate(fold_inds[k])

	result = []
	# begin cross validation procedure
	for miss in range(num_fold):
		valid_X = X[fold_inds[miss],:]
		valid_Y = Y[fold_inds[miss]]
		train_inds = np.concatenate([fold_inds[i] for i in range(num_fold) if i != miss])
		train_X = X[train_inds,:]
		train_Y = Y[train_inds]
		train = (train_X,train_Y,None)
		valid = (valid_X,valid_Y,None)
		arguments = params,train,valid
		result.append( run_train_valid(arguments) )
	return params, result

def free_cross_valid(args):
	params, train = args
	# svm_C,svm_gamma = params
	X,Y,T,S = train

	num_cv = FREE_CV_PARA[0]
	ratio = FREE_CV_PARA[1]
	# calculate the inds for each fold
	fold_inds = defaultdict(list)
	classes = np.unique(Y)
	for c in classes:
		inds = np.where(Y==c)[0]
		users = np.unique(T[inds])
		N_user = users.size
		max_shift_num = N_user % num_cv
		max_shift = int(N_user/num_cv)+1
		roll_size = [0] + [max_shift]*max_shift_num + [max_shift-1]*(num_cv-max_shift_num-1)
		splitter = round(ratio*N_user)
		# split users into num_fold groups
		for i in range(num_cv):
			users = np.roll(users,roll_size[i])
			for uid in users[0:splitter]:
				fold_inds[i,'train'].append( np.where(T == uid)[0] )
			for uid in users[splitter:]:
				fold_inds[i,'valid'].append( np.where(T == uid)[0] )

	for k in fold_inds:
		fold_inds[k] = np.concatenate(fold_inds[k])

	result = []
	# begin cross validation procedure
	for i in range(num_cv):
		train_inds = fold_inds[i,'train']
		train_X = X[train_inds,:]
		train_Y = Y[train_inds]
		train_T = T[train_inds]

		valid_inds = fold_inds[i,'valid']
		valid_X = X[valid_inds,:]
		valid_Y = Y[valid_inds]
		valid_T = T[valid_inds]

		train = (train_X,train_Y,train_T,None)
		valid = (valid_X,valid_Y,valid_T,None)
		arguments = params,train,valid
		result.append( run_train_valid(arguments) )

	return params, result

def sequence_accuracy(obj,typeset):
	X,Y_real,T,S = typeset
	uids = np.unique(T)
	# Y_pred = obj.predict(X)
	Y_pred = obj.decision_function(X)

	seq_acc = []
	for v in uids:
		selector = T == v
		pred = Y_pred[selector]
		real = Y_real[selector]
		# percent = np.sum(pred == real) / real.size
		# seq_acc.append((real[0],round(percent,3)))
		label = real[0]
		Z = np.sum(abs(pred))
		pred /= Z
		tmp = [-np.sum(pred[pred<0]), np.sum(pred[pred>0])]
		percent = tmp[int(label)]
		seq_acc.append((label,round(percent,3)))

	ret = []
	seq_acc = np.array(seq_acc)
	classes = np.unique(seq_acc[:,0])
	for c in classes:
		selector = seq_acc[:,0] == c
		segment_percent = seq_acc[selector,1]
		right_percent = segment_percent[segment_percent>0.5]
		false_percent = segment_percent[segment_percent<=0.5]
		N_total = np.sum(selector)
		N_right = right_percent.size
		ratio = round(N_right/N_total,3)
		if ratio < 1e-10:
			conf = 0.0
		else:
			conf = round(np.sum(right_percent)/N_right,3)
		if c > 0:
			name = 'E'
		else:
			name = 'N'
		if false_percent.size > 0:
			false_conf = round(np.mean(false_percent),3)
		else:
			false_conf = 0.0
		ret.append((name,N_right,ratio,conf, false_conf ))
	return ret

def run_evaluate_procedure(params,train,test):
	C,gamma = params
	obj = SVC(class_weight=CLASS_WEIGHT, C= C, gamma = gamma).fit(train[0],train[1])
	train_acc = round(obj.score(train[0],train[1]),3)
	test_acc = round(obj.score(test[0],test[1]),3)
	train_seq_acc = sequence_accuracy(obj,train)
	test_seq_acc = sequence_accuracy(obj,test)
	return (train_acc,test_acc, train_seq_acc,test_seq_acc)

def top_configuration(candidates):
	configs = []
	for k,param,s in candidates:
		if k not in configs:
			configs.append(k)

		if len(configs) > 10:
			return configs
	return configs

def f1_harmony(a,b,eps = 1e-5):
	if (a + b) < eps:
		score = 0.0
	else:
		score = (2*a*b)/(a+b)	
	return score

def get_score_new(cv_res):
	tmp = []
	for report in cv_res:
		info = report['seq_perf']
		novice = info[0]
		expert = info[1]
		a = f1_harmony(novice[2],expert[2])
		b = f1_harmony(novice[3],expert[3])
		tmp.append([a,b])
	ret = np.mean(tmp,axis=0)
	assert(ret.size == 2)
	score = ret[0] + 0.01 * ret[1]
	return score

def get_score(cv_res):
	tmp = []
	for report in cv_res:
		# a = report['0']['f1-score']
		# b = report['1']['f1-score']
		
		# info = report['seq_perf']
		# novice = info[0]
		# expert = info[1]
		# a = novice[2] * novice[3]
		# b = expert[2] * expert[3]

		# score = f1_harmony(a,b)

		score = report['accuracy']

		tmp.append( score )

	score = np.mean(tmp)
	# favor the stable cv performance
	# score -= 0.01*np.std(tmp)
	return score

STD_CV_FOLD = 4
FREE_CV_PARA = (8,0.8)
CLASS_WEIGHT = 'balanced'

INT =  []

def main(args):
	feature_data_file = args[0]
	norm_flag = int(args[1])
	split_seed = int(args[2])
	log = Logger('hypertune.log')
	log.verbose = True
	log.write("Input is {}".format(args))
	#########################
	split_ratio = 0.8
	cv_method = free_cross_valid
	svm_C = [1,10,100,1000,10000]
	svm_gamma = [b*10**p for p in range(-3,-1) for b in [1,3,5,7,9]]
	########################
	feature_data = utils.pickle_load(feature_data_file)
	if 'info' in feature_data:
		log.write(feature_data['info'])
		del feature_data['info']

	if norm_flag:
		log.write("Normalization is conducted!")
		for k,v in feature_data.items():
			x,y,t,s = v
			nx = mlib_data.normalize(x)
			feature_data[k] = (nx,y,t,s)

	# find all uids in the data
	key = list(feature_data.keys())[0]
	_,y,t,_ = feature_data[key]
	yt = np.vstack([t,y]).T
	userinfo = np.unique(yt,axis=0)
	userinfo = [tuple(row) for row in userinfo ]
	groups = split_users(userinfo,split_ratio,split_seed)

	if len(args) > 3:
		cv_result = utils.pickle_load(args[3])
	else:
		cv_result = {}
		svm_conf = list(utils.product([svm_C,svm_gamma]))
		counter = 0
		for k,v in feature_data.items():
			if len(INT) > 0 and  k not in INT:
				continue

			counter += 1
			print('Working on {} {}/{}'.format(k,counter,len(feature_data)))

			train, _ = split_into_groups(v,groups)
			p =  Pool(MAX_CORE)
			result = list(tqdm(p.imap_unordered(cv_method,zip(svm_conf,[train]*len(svm_conf))), total = len(svm_conf)))
			p.close(); p.join()
			for cf,res in result:
				cv_result[k,cf] = res
		utils.pickle_dump(cv_result, feature_data_file + ".res")

	# Evaluation procedure
	candidates = []
	for (k,cf), res in cv_result.items():
		score = get_score(res)
		candidates.append( (k,cf,score) )

	candidates.sort(key = lambda x:x[-1], reverse = True)
	log.write("Total number of items is {}".format(len(candidates)))
	log.write('Top HyperPara: {}'.format(top_configuration(candidates)))

	# only show top results
	for k,param,s in candidates[0:5]:
		data = feature_data[k]
		train, test =  split_into_groups(data,groups)
		result = run_evaluate_procedure(param,train,test)
		s = round(s,5)
		log.write('{} {} {} -> {}'.format(k,param,s,result))

if __name__ == '__main__':
	if len(sys.argv) < 4:
		print("Pass me args like:\n\t data_pkl_path normalization_flag split_seed [cv_result] [any other comments]!")
		exit(0)
	main(sys.argv[1:])