# coding: utf-8
import os,sys
import numpy as np
from multiprocessing import Pool
from collections import defaultdict

from pdb import set_trace
from pyLeon import utils
from pyLeon.misc import Logger

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from tune_hyper import mlib_data, construct_array_data, extract_feature, split_into_groups, split_users

def evluate_sequence_accuracy(obj,typeset):
	X,Y_real,T,S = typeset
	uids = np.unique(T)
	Y_pred = obj.predict(X)

	seq_acc = []
	for v in uids:
		selector = T == v
		pred = Y_pred[selector]
		real = Y_real[selector]
		percent = np.sum(pred == real) / real.size
		seq_acc.append((real[0],round(percent,3)))

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
			conf = ''
		else:
			conf = round(np.sum(right_percent)/N_right,3)
		if c > 0:
			name = 'E'
		else:
			name = 'N'
		if false_percent.size > 0:
			false_conf = round(np.mean(false_percent),3)
		else:
			false_conf = ''
		ret.append((name,N_right,ratio,conf, false_conf ))
	return ret

def run_exp_under_conf(params,X):
	lseg, ss, p_fea, C, gamma = params
	_,Y,T,S = construct_array_data(ori_data,lseg,ss)
	assert(X.shape[1] == p_fea)
	# X,_ = mlib_data.normalize(X)
	# X,_ = extract_feature(X,p_fea)
	# X,_ = mlib_data.normalize(X)
	train,test = split_into_groups((X,Y,T,S),groups)
	##############
	obj = SVC(class_weight=type_weight, C= C, gamma = gamma).fit(train[0],train[1])
	train_acc = round(obj.score(train[0],train[1]),3)
	test_acc = round(obj.score(test[0],test[1]),3)
	train_seq_acc = evluate_sequence_accuracy(obj,train)
	test_seq_acc = evluate_sequence_accuracy(obj,test)
	return (train_acc,test_acc, train_seq_acc,test_seq_acc)

def get_score(cv_res):
	tmp = []
	for report in cv_res:
		# score = 0.5*report['accuracy'] + 0.5*report['1']['f1-score']
		# score = 0.1*report['0']['f1-score'] + 0.9*report['1']['f1-score']
		# score = 0.5*report['1']['f1-score'] + 0.3*report['0']['f1-score'] + 0.2*report['accuracy']
		a = report['0']['f1-score']
		b = report['1']['f1-score']
		score = 2*a*b/(a+b)
		tmp.append( score )
	score = np.mean(tmp)# - 1 * np.std(tmp)
	return score

def analysis_result(data_file):
	global ori_data,groups,type_weight
	data = utils.pickle_load(data_file)
	gvars = data['globals']
	ori_data = mlib_data.load_data(gvars[0],binary = True)
	groups = split_users([(u,s) for u,m,s in ori_data], ratio = gvars[1], seed = gvars[2])
	print(gvars)

	if isinstance(gvars[-1],dict):
		print("Weight specified")
		type_weight = gvars[-1]
	else:
		print("No weight specified")
		type_weight = 'balanced'
	del data['globals']

	conf_score = []
	for data_conf,v in data.items():
		X = v['dataset']
		del v['dataset']
		for svm_conf,result in v.items():
			conf = data_conf+svm_conf
			score = get_score(result)
			conf_score.append((conf,score,X))
	conf_score.sort(key = lambda x:x[1], reverse = True)
	print('Total number of items: {}'.format(len(conf_score)))

	for c,s,X in conf_score[0:20]:
		result = run_exp_under_conf(c,X)
		print('{} <--> {}'.format(c,result))


	# top_confs = [item[0] for item in conf_score[0:20]]
	# top_res = Pool().map(run_exp_under_conf,top_confs)
	# for c,r in zip(top_confs,top_res):
	# 	print('{} <--> {}'.format(r,c))


if __name__ == '__main__':
	file_name = sys.argv[1]
	analysis_result(file_name)
