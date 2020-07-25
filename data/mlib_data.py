# coding: utf-8
import numpy as np
from collections import defaultdict
from copy import deepcopy
from pyLeon import utils
from pyLeon.potential import CLG
from pdb import set_trace

STEPSIZE = 15 # can be overwrite with an integer

def get_thresh(scores):
	X = scores.reshape(-1,1)
	_,b,s = CLG().fit(X).para 
	thresh = float(b + np.sqrt(s))
	return thresh

def load_data(fname,binary = True,thresh = 'auto'):
	raw = utils.pickle_load(fname)

	if binary == True and thresh == 'auto':
		allScore = np.array([s for u,m,s in raw])
		thresh = get_thresh(allScore)
		print("Auto thresh: {} ".format(thresh))
		print("Above thresh: {}".format(np.sum(allScore >= thresh)))

	ret = []
	for item in raw:
		uid,movdata,score = item
		if binary:
			score = int(score >= thresh)
		inds = list(range(0,movdata.shape[0],STEPSIZE))
		mov_new = movdata[inds,:]
		ret.append( (uid,mov_new,score) )

	return ret

def cut_series(data, time, shift):
	slice_size = int(time*90/STEPSIZE)
	shift_step = int(shift*90/STEPSIZE)
	ret = []
	# main logic
	for item in data:
		uid,movdata,score = item
		rows = movdata.shape[0]
		splitters = list(range(0,rows-slice_size+1,shift_step))
		for i,st in enumerate(splitters):
			ed = st + slice_size
			movslice = movdata[st:ed,:]
			ret.append((uid,movslice,score,i))
	return ret

def calculate_normal_const(data):
	norm_const = {}
	N,F = data.shape
	for f in range(F):
		mean = float(np.mean(data[:,f]))
		std = float(np.sqrt(np.var(data[:,f])))
		norm_const[f] = (mean,std)
	return norm_const

def normalize(data,norm_const = None):
	if norm_const is None:
		norm_const = calculate_normal_const(data)
	ret = deepcopy(data)
	N,F = data.shape
	for f in range(F):
		mean,std = norm_const[f]
		ret[:,f] = (data[:,f]-mean)/std
	return ret

def take_one_order(arr):
	R,C = arr.shape
	darr = np.zeros((R-1,C))
	for c in range(C):
		darr[:,c] = arr[1:,c] - arr[0:(R-1),c]
	return darr
