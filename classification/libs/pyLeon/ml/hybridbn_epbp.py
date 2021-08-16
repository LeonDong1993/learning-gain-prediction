# coding:utf-8
import numpy as np
from copy import deepcopy
from pyLeon.classes import *
from pyLeon.funcs import crossprod,progress
from pdb import set_trace

class HybridBN:
	def __init__(self,g,ev):
		self.g = g
		self.ev = ev
		self.potential = {}
		self.proposal  = {}

	def fit(self,traindata):
		for i in range(self.g.N):
			progress(i+1,self.g.N,"Learning")
			# learn the conditional distritbution
			node = self.g.V[i]
			parents = self.g.find_parents(i)
			discpar,contpar = self._group_parents(parents)
			config = self._get_configuration(discpar)
			pw = PotentialWrapper([i]+contpar,discpar)
			for c in config:
				subtrain = HybridBN.data_under_config(traindata,discpar,c)
				if node.attr == CONST.CONTINUOUS:
					pt = CLG()
				elif node.attr == CONST.DISCRETE:
					pt = SoftMax(node.domain)
				else:
					assert(False)
				pt.fit(subtrain[:,tuple([i]+contpar)])
				pw.add_potential(c,pt)
			self.potential[i] = pw
			# learn the proposal distribution
			if node.attr == CONST.CONTINUOUS:
				prop = CLG()
			elif node.attr == CONST.DISCRETE:
				prop = CPT([node])
			else:
				assert(False)
			prop.fit(traindata[:,i:i+1])
			self.proposal[i] = prop

		self.fg = self.g.factorize(potential = self.potential)
		for r in self.fg.rvs:
			node = self.fg.V[r]
			node.proposal = self.proposal[i]

	def predict(self,test_set):
		''' find MAP assignment for unknown R.V.
			test_set(np.array): each row is considered as a sample
		'''
		N,F = test_set.shape
		assert(F == self.g.N)
		pred_result = deepcopy(test_set)
		for n in range(N):
			progress(n+1,N,"Predicting")
			pred_result[n,:] = self._infer(test_set[n,:])
		return pred_result

	def _infer(self,testdata,it=0):
		# initilize evidence
		for r in self.ev:
			node = self.fg.V[r]
			node.value = testdata[r]

		solver = EPBPSolver(self.fg)
		solver.infer()
		prediction  = deepcopy(testdata)
		for r in self.fg.rvs:
			if r not in self.ev:
				prediction[r] = solver.MPE(r)
		return prediction

	def _group_parents(self,pids):
		discpar,contpar = [],[]
		for pid in pids:
			node = self.g.V[pid]
			if node.attr == CONST.CONTINUOUS:
				contpar.append(pid)
			else:
				assert(node.attr == CONST.DISCRETE)
				discpar.append(pid)
		return discpar,contpar

	def _get_configuration(self,pids):
		if len(pids) == 0:
			config = ['None']
		else:
			config = self.g.V[pids[0]].domain
			for i in pids[1:]:
				config = crossprod(config,self.g.V[i].domain)
		return config

	@staticmethod
	def data_under_config(data,pids,cfg):
		cfg = np.array(cfg)
		cfg = cfg.reshape(1,cfg.size)
		ret = deepcopy(data)
		for i,v in enumerate(pids):
			selector = ret[:,v] == cfg[0,i]
			ret = ret[selector,:]
		return ret

	@staticmethod
	def selftest():
		# Graph struture definition
		g = Graph(digraph = True)
		for i in range(7):
			node = Node()
			if i in [0,2,4,6]:
				node.attr = CONST.CONTINUOUS
			else:
				node.attr = CONST.DISCRETE
			g.add_vertice(node)

		E = [(0,3),(1,3),(1,4),(2,4),(3,5),(4,6)]
		for i,j in E:
			g.add_edge(i,j)

		g.V[1].domain = [0,1]
		g.V[3].domain = [1,3,5]
		g.V[5].domain = [3,6,9]

		# Random data generation
		N = 10000 ; test_ratio = 0.02
		cpt5 = CPT([g.V[5],g.V[3]])

		dataset = []
		for n in range(N):
			x0 = np.random.normal(0,10)
			x1 = np.random.choice(g.V[1].domain, p = [0.3,0.7])
			x2 = np.random.normal(50,5)
			if x1 == 1:
				para = np.array([-1,0,1])
			else:
				para = np.array([-0.2,0.1,0.4])
			tmp = np.exp(para*x0)
			dist = tmp/np.sum(tmp)
			x3 = np.random.choice(g.V[3].domain, p=dist)
			x4 = np.random.normal(x1+x2,10)
			x5 = cpt5.random(cond_val = [x3])
			x6 = np.random.normal(0.8*x4,5)
			dataset.append([x0,x1,x2,x3,x4,x5,x6])
		dataset = np.array(dataset)

		# split train and test data
		splitter = int(N*(1 - test_ratio))
		traindata = dataset[0:splitter,:]
		testdata = dataset[splitter:,:]

		hbn = HybridBN(g,[0,1,2,5,6])
		hbn.fit(traindata)
		# set_trace()
		test_num = int(N*test_ratio)
		prediction = hbn.predict(testdata)
		dis_rate = np.sum(prediction[:,3] == testdata[:,3])/test_num
		cont_diff = prediction[:,4] - testdata[:,4]
		rmse = np.sqrt(np.sum(cont_diff**2)/test_num)
		print('Result: {} {:.2f}'.format(dis_rate,rmse))
