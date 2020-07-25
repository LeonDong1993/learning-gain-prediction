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

from tune_hyper import gendata,mlib_data, split_into_groups, split_users, sequence_accuracy

def extract_user_segment(uid,sn,label,name,mov_type = 'learn'):
	user_movdata = gendata.load_user_motion(uid, mov_type, columns = 'full')
	header = user_movdata[0,:]
	target = user_movdata[1:,:]
	
	mlib_data.STEPSIZE = 1
	lseg,ss = best_conf[:2]
	seg_data = mlib_data.cut_series([(uid,target,'unknown')], lseg, ss)
	segment = seg_data[sn]
	motion = segment[1]
	# write into file
	fname = 'L{}_{}_{}'.format(label,uid,name)
	print('User {} #{} motion slice saved into {}, label {}'.format(uid,sn,fname,label))
	fh = open(fname,'w')
	print( ','.join(header) ,file = fh)
	for row in motion:
		print( ','.join(row) ,file = fh)
	fh.close()

def main(N):
	lseg, ss, p_fea, C, gamma = best_conf
	feature_data = utils.pickle_load(data_file)
	target_data = feature_data[lseg,ss,p_fea]
	X,Y,T,S  = target_data
	userinfo = np.unique(np.vstack([T,Y]),axis = 1)
	tmp = list(zip(userinfo[0,:],userinfo[1,:]))
	groups = split_users(tmp,ratio = split_ratio, seed = split_seed )
	train,test = split_into_groups((X,Y,T,S),groups)

	################
	obj = SVC(class_weight = type_weight, C = C, gamma = gamma, probability = True).fit(train[0],train[1])
	train_acc = obj.score(train[0],train[1])
	test_acc = obj.score(test[0],test[1])
	train_seq_acc = sequence_accuracy(obj,train)
	test_seq_acc = sequence_accuracy(obj,test)
	print(train_acc,test_acc, train_seq_acc,test_seq_acc)
	test_prob = obj.predict_proba(test[0])
	### extract segment
	einds = np.argsort(test_prob[:,1])[-N:]
	ninds = np.argsort(test_prob[:,0])[-N:]
	for i,eind in enumerate(einds):
		extract_user_segment(T[eind], S[eind], Y[eind],f'expert_{N-i}.movseg')
	
	for i,nind in enumerate(ninds):	
		extract_user_segment(T[nind], S[nind], Y[nind] ,f'novice_{N-i}.movseg')

if __name__ == '__main__':
	data_file = sys.argv[1]
	split_ratio = 0.8
	split_seed = 0
	type_weight = 'balanced'
	best_conf = (120, 7, 50, 10000, 0.07)
	main(25)
