# coding: utf-8
import time,random
from copy import deepcopy
from pdb import set_trace

class CONST:
	RV = 0
	FACTOR = 1
	DISCRETE = 2
	CONTINUOUS = 3

class MyObject:
	def __getitem__(self,key):
		return self.__dict__[key]

	def __setitem__(self,k,v):
		self.__dict__[k] = v

	def show(self):
		for k,v in self.__dict__.items():
			print('{} -> {}'.format(k,v))

class Logger:
	def __init__(self,logfile):
		self.file ={'default':logfile}
		self.verbose = True
		self.write('--> NEW <--')

	def write(self,msg,logtype='default'):
		if logtype not in self.file:
			tmp = deepcopy(self.file['default'])
			tmp=tmp.split('.')
			fpath = tmp[0]
			for s in tmp[1:-1]:
				fpath += "."+s
			fpath += "_" +logtype+ "."+tmp[-1]
			self.file[logtype]=fpath

		if self.verbose:
			print(msg)

		cur_time = Timer.get_time()
		fh=open(self.file[logtype],'a')
		fh.write(cur_time + '---' + msg + "\n")
		fh.close()

class Timer:
	@staticmethod
	def get_time():
		return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

	@staticmethod
	def sleep(low,high=None, verbose = False):
		if high == None:
			secs=low
		else:
			secs=random.randint(low,high)
		if verbose:
			print(f"Sleep for {secs} seconds....\r")
		time.sleep(secs)
		return
