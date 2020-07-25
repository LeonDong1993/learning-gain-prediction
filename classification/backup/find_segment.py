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

from tune_hyper import gendata, mlib_data, construct_array_data, extract_feature, split_into_groups, split_users

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
			conf = 'N/A'
		else:
			conf = round(np.sum(right_percent)/N_right,3)
		if c > 0:
			name = 'E'
		else:
			name = 'N'
		if false_percent.size > 0:
			false_conf = np.mean(false_percent)
		else:
			false_conf = 'N/A'
		ret.append((name,N_right,ratio,conf, false_conf ))
	return ret

def extract_user_segment(uid,sn,label,name):
	mov_type = data_file.split('_')[0]
	print('Motion type is {}'.format(mov_type))
	user_movdata = gendata.load_user_motion(uid, mov_type, columns = 'full')
	header = user_movdata[0,:]
	target = user_movdata[1:,:]

	mlib_data.STEPSIZE = 1
	lseg,ss = best_conf[:2]
	seg_data = mlib_data.cut_series([(uid,target,'unknown')], lseg, ss)
	segment = seg_data[sn]
	motion = segment[1]
	# write into file
	fname = '{}_{}'.format(uid,name)
	print('User {} #{} motion slice saved into {}, label {}'.format(uid,sn,fname,label))
	fh = open(fname,'w')
	print( ','.join(header) ,file = fh)
	for row in motion:
		print( ','.join(row) ,file = fh)
	fh.close()

def main(N):
	lseg, ss, p_fea, C, gamma = best_conf
	ori_data = mlib_data.load_data(data_file, binary = True)
	groups = split_users([(u,s) for u,m,s in ori_data],ratio = split_ratio, seed = split_seed )
	#################
	X,Y,T,S = construct_array_data(ori_data,lseg,ss)
	X,_ = mlib_data.normalize(X)
	X,_ = extract_feature(X,p_fea)
	X,_ = mlib_data.normalize(X)
	train,test = split_into_groups((X,Y,T,S),groups)
	################
	obj = SVC(class_weight = type_weight, C = C, gamma = gamma, probability = True).fit(train[0],train[1])
	train_acc = obj.score(train[0],train[1])
	test_acc = obj.score(test[0],test[1])
	train_seq_acc = evluate_sequence_accuracy(obj,train)
	test_seq_acc = evluate_sequence_accuracy(obj,test)
	print(train_acc,test_acc, train_seq_acc,test_seq_acc)
	test_prob = obj.predict_proba(test[0])
	set_trace()
	### extract segment
	einds = np.argsort(test_prob[:,1])[-N:]
	ninds = np.argsort(test_prob[:,0])[-N:]
	for i,eind in enumerate(einds):
		extract_user_segment(T[eind], S[eind], Y[eind],f'expert_{N-i}.csv')
	
	for i,nind in enumerate(ninds):	
		extract_user_segment(T[nind], S[nind], Y[nind] ,f'novice_{N-i}.csv')

if __name__ == '__main__':
	data_file = 'learn_score'
	split_ratio = 0.8
	split_seed = 5
	type_weight = {0: 0.1, 1: 0.9}
	best_conf = (60, 40, 70, 100, 0.004)
	main(1)
