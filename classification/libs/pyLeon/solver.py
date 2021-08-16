import numpy as np
from copy import deepcopy
from .utils import product,progress
from .misc import CONST
from .potential import CLG
from pdb import set_trace

class EPBP:
	def __init__(self,fg,sn=100):
		""" init the expectation particle belief propgation solver for Hybrid PGM
		Args:
			fg (Graph ojbect): the input factor graph
			sn (int): number of samples used for solver
		"""
		self.fg = fg
		self.sample_num = sn
		self.var_thresh = 0.2

	def _update_sample(self,init=False):
		for r in self.fg.rvs:
			node = self.fg.V[r]
			if not init:
				node.old_sample = node.sample

			if node.value == None:
				if node.attr == CONST.DISCRETE:
					node.sample = deepcopy(node.domain)
				else:
					node.sample = node.proposal.random(self.sample_num)
			else:
				node.sample = [node.value]

	def _msgr2f(self,rid,fid,sample):
		node = self.fg.V[rid]
		m = 1.0
		if node.value == None:
			for f in node.nb:
				if f != fid:
					m *= self.M[f,rid][sample]
			m /= node.proposal.prob(sample)
		return m

	def _msgf2r(self,fid,rid,sample):
		# calculate the join domain
		factor = self.fg.V[fid]
		args = []
		for r in factor.nb:
			if r == rid:
				args.append([sample])
			else:
				args.append(self.fg.V[r].old_sample)
		domain = product(args)

		# max-product BP
		ret = 0.0
		for X in domain:
			m = 1.0
			for i,r in enumerate(factor.nb):
				if r != rid:
					m *= self.M[r,fid][X[i]]
			valx = dict(zip(factor.nb,X))
			m *= factor.potential.prob(valx)
			ret = max(ret,m)
		return ret

	def _update_proposal(self,rid):
		node = self.fg.V[rid]
		args = []
		for f in node.nb:
			# project the msg into a gaussian
			message = self.M[f,rid]
			first_moment = 0.0
			second_moment = 0.0
			z = sum(message.values())
			for k,v in message.items():
				first_moment += k*v/z
				second_moment += k*k*v/z
			mu = first_moment
			var = max(second_moment - mu*mu,self.var_thresh)
			args.append((mu,var))

		mu,S = CLG.product_gaussian(args)
		S = max(S,self.var_thresh)
		proposal = CLG()
		proposal.para = [0,mu,S]
		return proposal

	@staticmethod
	def normalize_message(msg):
		m = deepcopy(msg)
		z = sum(msg.values())
		for k in msg:
			m[k] /= z
		return m

	def infer(self,it=0):
		""" run the loppy EPBP on the given factor graph
		Args:
			it (int): number of iterations, default is the number of node in factor graph
		"""
		fg = self.fg  #alias
		if it==0: it = fg.N
		self.M = {}

		# initialize message and samples
		self._update_sample(init=True)
		for r in fg.rvs:
			node = fg.V[r]
			msg = dict(zip(node.sample, [1.0 for s in node.sample]))
			for f in node.nb:
				self.M[r,f] = deepcopy(msg)
				self.M[f,r] = deepcopy(msg)

		for _ in range(it):
			# pass message from rv to factor
			for r in fg.rvs:
				node = fg.V[r]
				for f in node.nb:
					msg = {}
					for s in node.sample:
						msg[s] = self._msgr2f(r,f,s)
					self.M[r,f] = EPBP.normalize_message(msg)

			# update proposal and sample
			for i in fg.rvs:
				node = fg.V[i]
				if node.attr == CONST.CONTINUOUS:
					node.proposal = self._update_proposal(i)
			self._update_sample()

			# pass message from factor to rv, new samples used
			for f in fg.factors:
				factor = fg.V[f]
				for r in factor.nb:
					node = fg.V[r]
					if node.value == None:
						msg = {}
						for s in node.sample:
							msg[s] = self._msgf2r(f,r,s)
						self.M[f,r] = EPBP.normalize_message(msg)

	def MPE(self,rid):
		node = self.fg.V[rid]
		assert(node.value == None)
		max_assignment = None
		maxval = 0.0
		for s in node.sample:
			val = 1.0
			for f in node.nb:
				val *= self.M[f,rid][s]
			if val > maxval:
				max_assignment = s
				maxval = val
		return max_assignment

class GaBP:
	def __init__(self,fg):
		self.fg = fg

	def _prod_inmsg(self,rid):
		rv = self.fg.V[rid]
		arr = np.array([self.M[f,rid] for f in rv.nb])
		vec_p = arr[:,1]
		vec_mu = arr[:,0]
		X = sum(vec_mu*vec_p)
		P = sum(vec_p)
		return X,P
		
	def _msgr2f(self,rid,fid,prod_msg):
		X,P = prod_msg
		mu,p = self.M[fid,rid]
		P -= p
		X -= mu*p

		if P == 0:
			msg = (1.0,1e+8)
		else:
			msg = (X/P,P)
		return msg

	def _msgf2r(self,fid,rid):
		factor = self.fg.V[fid]
		rv = self.fg.V[rid]
		if rv.value is None:
			a,b,s = factor.potential.para
			vec_a = a.flatten()
			domain = factor.potential.domain
			# construct coeff vector 
			c = [0]*len(domain)
			for i,nid in enumerate(factor.nb):
				ind = domain.index(nid)
				if ind == 0:
					c[i] = -1
				else:
					c[i] = vec_a[ind-1]
			
			P = 1.0/s
			mu = [self.M[i,fid][0] for i in factor.nb]
			p = [self.M[i,fid][1] for i in factor.nb]
			ind = factor.nb.index(rid)

			if len(factor.nb) == 1:
				msg = (b,P)
			else:
				# calculate mu
				msg_mu = -b/c[ind]
				for j,nid in enumerate(factor.nb):
					if nid!=rid:
						msg_mu += -c[j]*mu[j]/c[ind]
				# calculate P
				product = np.prod(p)
				numerator = P*c[ind]*c[ind]*product/p[ind]
				denominator = product/p[ind]
				for j,nid in enumerate(factor.nb):
					if nid!=rid:
						denominator += P*c[j]*c[j]*product/(p[ind]*p[j])
				msg_P = numerator/denominator
				msg = (msg_mu,msg_P)
		else:
			msg = (None,None)
		return msg


	def infer(self,it = 0):
		fg = self.fg # alias
		if it<=0: it=int(0.5*fg.N)
		self.M = {}
		# initialize all the messages
		for i,j in fg.get_edges():
			self.M[i,j] = (1.0,1.0)
		
		# initialize all evidence msg that nerver upadate in propagation
		for r in fg.rvs:
			rv = fg.V[r]
			if rv.value != None:
				for f in rv.nb:
					self.M[r,f] = (rv.value,1e+8)

		# start loppy GaBP
		for _ in range(it):
			# progress(xx+1,it)
			# passing message from rv to factor
			for r in fg.rvs:
				rv = fg.V[r]
				if rv.value is None:
					prod_msg = self._prod_inmsg(r)
					for f in rv.nb:
						self.M[r,f] = self._msgr2f(r,f,prod_msg)

			# passing message from factor to rv
			for f in fg.factors:
				factor = fg.V[f]
				for r in factor.nb:
					self.M[f,r] = self._msgf2r(f,r)
		return self

	def MPE(self,nid,conf = False):
		assert(nid in self.fg.rvs)
		rv = self.fg.V[nid]

		if rv.value is None:
			X,P = self._prod_inmsg(nid)
			mpe = X/P
			prob = P
		else:
			mpe = rv.value
			prob = 1.0
		
		if conf:
			ret = (mpe,prob)
		else:
			ret = mpe
		return ret