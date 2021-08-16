# coding:utf-8
import numpy as np
from copy import deepcopy
from functools import partial
from pyLeon.graph import Node,Graph
from pyLeon.utils import crossprod,allin,progress
from pyLeon.potential import CLG
from pyLeon.solver import GaBP
from pyLeon.misc import MyObject
from pdb import set_trace

class Potential(MyObject):
	def __repr__(self):
		return 'ids:{} P:{}'.format(self.ids,self.P)

class DiscreteRV(Node):
	def __init__(self,desc,domain):
		self.desc = desc
		self.domain = domain

class TreeBN:
	def __init__(self,g,ev):
		assert(isinstance(g,Graph)), "Input should be a Graph instance"
		assert(g.digraph), "Only directed graph allowed"

		for i in range(g.N):
			node = g.V[i]
			parents = g.find_parents(i)
			assert(isinstance(node,DiscreteRV)), "Vertice should be instance of DiscreteRV"
			assert(len(parents) <=1), "At most one parent is allowed for each node"

		self.graph = g
		self.ev = ev

	@staticmethod
	def chowliu_tree(data):
		'''
		Learn a chowliu tree structure based on give data
		data: S*N numpy array, where S is #samples, N is #RV (Discrete)
		'''
		S,D = data.shape
		marginals = {}
		# compute single r.v. marginals
		totalnum = D + (D*(D-1))/2
		nownum = 0
		for i in range(D):
			nownum += 1; progress(nownum,totalnum,'Learning chowliu tree')
			values, counts = np.unique(data[:,i], return_counts=True)
			marginals[i] = dict(zip(values, counts))

		# compute joint marginal for each pair
		for i,j in crossprod(range(D),range(D)):
			nownum += 1; progress(nownum,totalnum,'Learning chowliu tree')
			values, counts = np.unique(data[:,(i,j)], axis=0 ,return_counts=True)
			values = list(map(lambda x:tuple(x),values))
			marginals[i,j] = dict(zip(values, counts))
			allcomb = crossprod(list(marginals[i].keys()),list(marginals[j].keys()),'full')
			for v in allcomb:
				if v not in marginals[i,j]: marginals[i,j][v] = 0

		# normalize all marginals
		for key in marginals:
			dist = marginals[key]
			summation = sum(dist.values())
			for k in dist: dist[k] = (dist[k]+1) / float(summation) # 1- correction

		mutual = {}
		# compute mutual information
		for i,j in crossprod(range(D),range(D)):
			mutual[i,j] = 0
			for vi,vj in marginals[i,j]:
				mutual[i,j] += np.log(marginals[i,j][vi,vj] / (marginals[i][vi] * marginals[j][vj])) * marginals[i,j][vi,vj]

		# find the maximum spanning tree
		G = Graph(digraph=False)
		for i in range(D):
			node = DiscreteRV(desc = 'N{}'.format(i), domain = list(marginals[i].keys()))
			G.add_vertice(node)

		for i,j in mutual:
			G.add_edge(i,j,weight = mutual[i,j])

		G = G.max_spanning_tree()
		root = int(D/2)
		G = G.todirect(root)
		return G

	def fit(self,traindata):
		'''
		MLE learning, basically empirical distribution
		traindata: S*N numpy array, where S is #samples, N is #RV (Discrete)
		'''
		N,D = traindata.shape
		assert(self.graph.N == D), "Input data not valid"

		self.cpt = {}
		for i in range(self.graph.N):
			domain = self.graph.V[i].domain
			parents = self.graph.find_parents(i)

			if len(parents) == 0: # root node
				# learn the node potential
				values, counts = np.unique(traindata[:,i], return_counts=True)
				dist = dict(zip(values, counts))
				for v in domain:
					if v not in dist: dist[v] = 1 # 1-correction
				# normalize
				summation = sum(dist.values())
				for k in dist:dist[k] /= float(summation)
				self.cpt[i] = dist

			else:
				# create uniform node potential
				self.cpt[i] = dict(zip(domain, [1]*len(domain) ))
				# learn the edge potential
				dist = {}
				assert(len(parents) == 1), "Each vertice can only have at most one parent!"
				j = parents[0]
				jdomain = self.graph.V[j].domain
				values, counts = np.unique(traindata[:,(i,j)], axis=0 ,return_counts=True)
				values = list(map(lambda x:tuple(x),values))
				dist= dict(zip(values, counts))
				allcomb = crossprod(domain,jdomain,flag = 'full')
				for v in allcomb:
					if v not in dist: dist[v] = 1 #1-correction
				# normalize
				for vj in jdomain:
					summation = sum(map(lambda vi:dist[vi,vj],domain))
					for vi in domain:
						dist[vi,vj] /= float(summation)
				self.cpt[i,j] = dist

	def predict(self,testdata):
		'''
		predict the values of non-evidence RV given the value of evidence RV
		testdata: 1*N list / numpy array, N is #RV
		'''

		# first, from leaves to root
		order = self.graph.toposort(reverse = True)
		message = {}
		for i in order:
			parents = self.graph.find_parents(i)
			children = self.graph.find_children(i)

			if len(parents) == 0:
				continue
			msg = {}
			assert(len(parents) == 1)
			j = parents[0]
			if j in self.ev:
				continue

			ivals = self.graph.V[i].domain
			jvals = self.graph.V[j].domain

			if i not in self.ev:
				for vj in jvals:
					msg[vj] = []
					for vi in ivals:
						production = 1.0
						for c in children:
							production *= message[c,i][vi]
						msg[vj].append(self.cpt[i][vi] * self.cpt[i,j][vi,vj] * production)
					msg[vj] = max(msg[vj])
			else:
				for vj in jvals:
					msg[vj] = self.cpt[i,j][testdata[i], vj]

			# normalize message
			summation = sum(msg.values())
			for k in msg: msg[k] /= float(summation)
			message[i,j] = msg

		# second, from root to leaves
		order.reverse()
		for i in order:
			parents = self.graph.find_parents(i)
			children = self.graph.find_children(i)
			ivals = self.graph.V[i].domain

			for j in children:
				if j in self.ev:
					continue
				jvals = self.graph.V[j].domain

				msg = {}
				if i not in self.ev:
					for vj in jvals:
						msg[vj] = []
						for vi in ivals:
							production = 1.0
							for p in parents:
								production *= message[p,i][vi]
							for c in children:
								if c == j:continue
								production *= message[c,i][vi]
							msg[vj].append(self.cpt[i][vi] * self.cpt[j,i][vj,vi] * production)
						msg[vj] = max(msg[vj])
				else:
					for vj in jvals:
						msg[vj] = self.cpt[j,i][vj,testdata[i]]
				# normalize message
				summation = sum(msg.values())
				for k in msg: msg[k] /= float(summation)
				message[i,j] = msg

		# calculate node belief
		prediction = deepcopy(testdata)
		for i in range(self.graph.N):
			if i not in self.ev:
				belief = {}
				parents = self.graph.find_parents(i)
				children = self.graph.find_children(i)
				nodes = parents + children
				for v in self.graph.V[i].domain:
					belief[v] = self.cpt[i][v]
					for n in nodes:
						belief[v] *= message[n,i][v]
				prediction[i] = max(belief.items(),key= lambda x:x[1])[0]

		return prediction

class DBN:
	def __init__(self,g,ev,intercnt='default'):
		'''
		intercnt - the connection between two time slice, 'default' means we have 1-to-1 connection between
		all the state variables.
		'''
		assert(isinstance(g,Graph)), "Input should be a Graph instance"
		assert(g.digraph), "Only directed graph allowed"

		for i in range(g.N):
			node = g.V[i]
			assert(isinstance(node,DiscreteRV)), "Vertice should be instance of DiscreteRV"

		sv = list(filter(lambda x:x not in ev,range(g.N)))

		self.G = g
		self.EV = ev
		self.SV = sv
		self.CPT = {}
		self.ICPT = {}
		self.filter = partial(self.smooth,smooth = False)
		self.predict = partial(self.smooth,smooth = False)

		# construt two time slice graph G2
		mapping = {}
		G2 = deepcopy(self.G)

		for i in self.SV:
			node = deepcopy(self.G.V[i])
			node.desc = 'f-' + node.desc
			nid = G2.add_vertice(node)
			mapping[i]=nid

		if intercnt == 'default':
			intercnt = [(i,i) for i in self.SV]

		for i,j in intercnt:
			G2.add_edge(mapping[i],j)

		self.G2 = G2
		self.M = mapping

		# add edges in G
		edges = self.G.get_edges()
		for i,j in edges:
			if (i in self.SV) and (j in self.SV):
				a,b = self.M[i],self.M[j]
				self.G2.add_edge(a,b)

		# reverse mapping used for backward message passing
		self.rM = {}
		for k,v in self.M.items():
			self.rM[v] = k

		# init node CPT
		for i in self.SV:
			parents = self.G.find_parents(i)
			self.ICPT[i] = self.init_CPT(i,parents)

		for i in range(self.G.N):
			parents = self.G2.find_parents(i)
			self.CPT[i] = self.init_CPT(i,parents)

	def init_CPT(self,i,parents):
		cpt = Potential()
		ids = [i] + parents
		table_size = tuple(map(lambda x: len(self.G2.V[x].domain) ,ids))
		cpt.ids = ids
		cpt.P = np.zeros(table_size)
		return cpt

	def min_clique(self,G,ids):
		# find the minimum clique in G contains all id in ids
		candidates = []
		for i in range(G.N):
			nids = G.V[i].ids
			if allin(ids,nids):
				candidates.append( (i,len(nids)) )
		best = min(candidates,key=lambda x:x[1])
		return best[0]

	def init_potential(self,ids):
		potential = Potential()
		table_size = tuple(map(lambda x: len(self.G2.V[x].domain) ,ids))
		potential.ids = ids
		potential.P = np.ones(table_size)
		return potential

	def init_message(self,G,ret):
		for i in range(G.N):
			clique = G.V[i]
			ret[i] = self.init_potential(clique.ids)
		return

	def multiply_potential(self,p1,p2):
		if p1.ids == p2.ids:
			newp = self.init_potential(p1.ids)
			newp.P = p1.P * p2.P
			return newp

		if len(p1.ids) >= len(p2.ids):
			pb = p1; ps = p2
		else:
			pb = p2; ps = p1
		assert(allin(ps.ids,pb.ids))

		pt = deepcopy(pb)
		for npi,npv in np.ndenumerate(ps.P):
			idx = []
			for v in pt.ids:
				if v in ps.ids:
					idx.append( npi[ps.ids.index(v)] )
				else:
					idx.append( slice(None) )
			idx = tuple(idx)
			pt.P[idx] *= npv

		pt.P = pt.P/np.sum(pt.P)
		return pt

	def multiply_CPT(self,G,E,ret,init=False):
		for i in range(G.N):
			clique = G.V[i]
			assert(ret[i].ids == clique.ids)
			for eid in clique.elim:
				if eid in self.SV:
					if init:
						ret[i] = self.multiply_potential(ret[i],self.ICPT[eid])
					else:
						ret[i] = self.multiply_potential(ret[i],self.CPT[eid])
				if eid in self.EV:
					ret[i] = self.multiply_potential(ret[i],self.CPT[eid])

			# condition out the evidence variable
			newids = list(filter(lambda x:x not in self.EV,clique.ids))
			if len(newids) < len(clique.ids):
				potential = self.init_potential(newids)
				for npi,npv in np.ndenumerate(potential.P):
					idx = [-1 for v in clique.ids]
					for si,v in enumerate(clique.ids):
						if v in self.EV:
							idx[si] = E[v]
						else:
							idx[si] = npi[newids.index(v)]
					idx = tuple(idx)
					potential.P[npi] = ret[i].P[idx]
				ret[i] = potential

	def marginalize(self,pt,ids):
		if pt.ids == ids:
			newp = deepcopy(pt)
			newp.P = newp.P/np.sum(newp.P)
			return newp

		newp = deepcopy(pt)
		sumout = list(filter(lambda v:v not in ids,pt.ids))
		for s in sumout:
			dim = newp.ids.index(s)
			newp.P = np.amax(newp.P,axis = dim)
			newp.ids.remove(s)
		return newp

	def get_message(self,p1,p2,timestep = 0):
		assert(timestep in [-1,0,1])
		# get the message pass from p1 -> p2
		ids = []
		if timestep > 0:
			for i in p1.ids:
				assert(i not in self.EV)
				if (i in self.SV) and (self.M[i] in p2.ids): ids.append(i)

		elif timestep <0:
			for i in p1.ids:
				assert(i not in self.EV)
				if (i not in self.SV) and (self.rM[i] in p2.ids): ids.append(i)

		else:
			ids = list(filter(lambda x:x in p2.ids,p1.ids))

		msg = self.marginalize(p1,ids)

		if timestep > 0:
			msg.ids = list(map(lambda x:self.M[x],msg.ids))
		if timestep < 0:
			msg.ids = list(map(lambda x:self.rM[x],msg.ids))

		return msg

	def calculate_msg(self,G,npt):
		message = {}
		g = G.todirect(0)
		order = g.toposort(reverse = True)
		# do message passing
		for i in order:
			parents = g.find_parents(i)
			children = g.find_children(i)

			if len(parents) == 0: continue
			assert(len(parents) == 1)
			j = parents[0]

			msg = npt[i]
			for c in children:
				msg = self.multiply_potential(msg,message[c,i])
			message[i,j] = self.get_message(msg,npt[j])

		# from root to leaves
		order.reverse()
		for i in order:
			parents = g.find_parents(i)
			children = g.find_children(i)
			for j in children:
				msg = npt[i]
				for p in parents:
					msg = self.multiply_potential(msg,message[p,i])
				for c in children:
					if c == j: continue
					msg = self.multiply_potential(msg,message[c,i])
				message[i,j] = self.get_message(msg,npt[j])
		return message

	def collect_msg(self,G,r,npt,msg):
		neignbors = G.find_neighbor(r)
		pt = npt[r]
		for n in neignbors:
			pt = self.multiply_potential(pt,msg[n,r])
		return pt

	def smooth(self,data,numnodes=4,smooth=True):
		assert(numnodes > 1)
		st = 0
		appro = []
		while st < len(self.SV):
			ed = st + numnodes
			if ed > len(self.SV):
				ed = len(self.SV)
			appro.append(self.SV[st:ed])
			st = ed

		# create junction tree J1
		T1G = deepcopy(self.G)
		T1G = T1G.moralize()
		for bkc in appro:
			for s,t in crossprod(bkc,bkc):
				T1G.add_edge(s,t)

		self.J1 = T1G.junction_tree(preserve=self.G)

		# find come and out node
		self.J1.out = []
		for bkc in appro:
			self.J1.out.append( self.min_clique(self.J1,bkc) )
		self.J1.come = deepcopy(self.J1.out)

		# create junction tree Jt
		T2G = self.G2.moralize()
		for bkc in appro:
			for s,t in crossprod(bkc,bkc):
				T2G.add_edge(s,t)

			fbkc = list(map(lambda x:self.M[x],bkc))
			for s,t in crossprod(fbkc,fbkc):
				T2G.add_edge(s,t)

		self.J2 = T2G.junction_tree(preserve = self.G2)

		# find come and out node
		self.J2.out = []
		for bkc in appro:
			self.J2.out.append( self.min_clique(self.J2,bkc) )

		self.J2.come = []
		for bkc in appro:
			fbkc = list(map(lambda x:self.M[x],bkc))
			self.J2.come.append( self.min_clique(self.J2,fbkc) )


		T,N = data.shape
		assert(N == self.G.N)

		fmsg = {}
		for t in range(T):
			progress(t+1,T, 'Forward')

			fmsg[t] = {}
			evidence = data[t,:]

			if t==0:
				self.init_message(self.J1,fmsg[t])
				self.multiply_CPT(self.J1,evidence,fmsg[t],init=True)
				# collect message to out node for each bk cluster
				npt = deepcopy(fmsg[t])
				message = self.calculate_msg(self.J1,npt)
				for i in self.J1.out:
					fmsg[t][i] = self.collect_msg(self.J1,i,npt,message)

			else:
				pt = t-1
				self.init_message(self.J2,fmsg[t])
				self.multiply_CPT(self.J2,evidence,fmsg[t])
				# absorb message from the previous time slice
				for i,inid in enumerate(self.J2.come):
					if pt == 0:
						outid = self.J1.out[i]
					else:
						outid = self.J2.out[i]

					msg = self.get_message(fmsg[pt][outid],fmsg[t][inid],timestep = 1)
					fmsg[pt][outid,-1] = msg
					fmsg[t][inid] = self.multiply_potential(msg,fmsg[t][inid])

				npt = deepcopy(fmsg[t])
				message = self.calculate_msg(self.J2,npt)
				for i in self.J2.out:
					fmsg[t][i] = self.collect_msg(self.J2,i,npt,message)

			if t==(T-1):
				for i,outid in enumerate(self.J2.out):
					inid = self.J2.come[i]
					fmsg[t][outid,-1] = self.get_message(fmsg[t][outid],fmsg[t][inid],timestep = 1)

		if smooth:
			endtime = -1
		else:
			endtime = T

		bmsg = {}
		for t in range(T-1,endtime,-1):
			progress(T-t,T, 'Backward')

			bmsg[t] = {}
			evidence = data[t,:]

			if t==(T-1):
				curG = self.J2
				self.init_message(curG,bmsg[t])
				self.multiply_CPT(curG,evidence,bmsg[t])
				npt = deepcopy(bmsg[t])
				message = self.calculate_msg(curG,npt)
				for i,inid in enumerate(curG.come):
					bmsg[t][inid] = self.collect_msg(curG,inid,npt,message)
					outid = curG.out[i]
					bmsg[t][-1,outid] = self.init_potential(appro[i])

			if t<(T-1):
				nt = t+1
				curG = self.J2
				if t==0:
					curG = self.J1
				# initialize message
				self.init_message(curG,bmsg[t])
				if t==0:
					self.multiply_CPT(curG,evidence,bmsg[t],init=True)
				else:
					self.multiply_CPT(curG,evidence,bmsg[t])
				# absorb message from the previous time slice
				for i,outid in enumerate(curG.out):
					inid = self.J2.come[i]
					msg = self.get_message(bmsg[nt][inid],bmsg[t][outid],timestep = -1)
					bmsg[t][-1,outid] = msg
					bmsg[t][outid] = self.multiply_potential(msg,bmsg[t][outid])

				npt = deepcopy(bmsg[t])
				message = self.calculate_msg(curG,npt)
				for i in curG.come:
					bmsg[t][i] = self.collect_msg(curG,i,npt,message)


		prediction = deepcopy(data)
		for t in range(T):
			if t==0:
				tg = self.J1
			else:
				tg = self.J2

			for bki,outid in enumerate(tg.out):
				fP = fmsg[t][outid,-1]
				fP.ids = list(map(lambda x:self.rM[x],fP.ids))
				potential = fP
				if smooth:
					bP = bmsg[t][-1,outid]
					potential =  self.multiply_potential(potential,bP)
				P = potential.P/np.sum(potential.P)
				idx = np.unravel_index(P.argmax(), P.shape)
				for v in appro[bki]:
					prediction[t,v] = idx[fP.ids.index(v)]

		return prediction

	def get_domain(self,nids):
		n0 = self.G2.V[nids[0]]
		D = n0.domain
		for i in nids[1:]:
			node = self.G2.V[i]
			D = crossprod(D,node.domain,flag='full')
		return D

	def norm_CPT(self,cpt):
		ratio = 1e-3
		X = np.sum(cpt.P)
		addv = int(X*ratio)
		addv += int(addv==0)
		cpt.P += addv
		newP = deepcopy(cpt.P)

		if len(cpt.ids) == 1:
			newP = newP/np.sum(cpt.P)
		else:
			n = cpt.ids[0]
			domain = self.get_domain(cpt.ids[1:])
			for v in domain:
				if not isinstance(v,tuple): v = tuple([v])
				index = tuple([self.G.V[n].domain]) + v
				summation = np.sum(cpt.P[index])
				assert( summation!=0 )
				newP[index] /= summation
		# return
		cpt.P = newP

	def fit(self,traindata):
		# traindata - list of 2D numpy array
		M = len(traindata)
		for i in range(M):
			progress(i+1,M,'DBN learning')
			data = traindata[i]
			T,N = data.shape
			assert(N == self.G.N)
			# basically learning the empirical distribution
			for t in range(T):
				now = data[t,:]
				if t == 0:
					for i in self.SV:
						idx = tuple(now[self.ICPT[i].ids])
						self.ICPT[i].P[idx] += 1
				else:
					prev = data[t-1,:]
					exnow = np.append(now,[0 for i in self.SV])
					for k,v in self.M.items():
						exnow[v] = prev[k]

					for i in self.SV:
						idx = tuple(exnow[self.CPT[i].ids])
						self.CPT[i].P[idx] += 1

				for i in self.EV:
					idx = tuple(now[self.CPT[i].ids])
					self.CPT[i].P[idx] += 1

		# normalize all CPT
		for i in range(self.G.N):
			self.norm_CPT(self.CPT[i])

		for i in self.SV:
			self.norm_CPT(self.ICPT[i])
		return

class GBN:
	def __init__(self,g,ev):
		assert(isinstance(g,Graph)), "Input should be a Graph instance"
		assert(g.digraph), "Only directed graph allowed"
		for i in range(g.N):
			parents = g.find_parents(i)
			assert(len(parents) <=1 ), "At most one parent is allowed for each node"
		self.g = g
		self.ev = ev
		self.potential = {}

	def fit(self,traindata):
		N,D = traindata.shape
		assert(self.g.N == D), "Input data not valid"

		self.dist= {}
		for i in range(self.g.N):
			parents = self.g.find_parents(i)

			if len(parents) == 0:
				# learn the node potential
				mu = np.mean(traindata[:,i])
				P = 1.0/np.var(traindata[:,i])
				self.dist[i] = (mu,P)
				self.root = i
				clg = CLG()
				clg.para = (np.array([0]), mu, 1.0/P)
				clg.domain = [i]
				self.potential[i] = clg
			else:
				j = parents[0]
				mu = np.mean(traindata[:,(i,j)],axis=0)
				sigma = np.cov(traindata[:,(i,j)],rowvar=False)
				# Linear gaussian, mu=b0+b1*y
				b1 = sigma[0,1]/sigma[1,1]
				b0 = mu[0] - b1 * mu[1]
				sig = sigma[0,0] - sigma[0,1]*sigma[1,0]/sigma[1,1]
				P = 1.0/sig
				clg = CLG()
				clg.para = (np.array([b1]),b0,1.0/P)
				clg.domain = [i,j]
				self.potential[i] = clg
				self.dist[i,j] = (b0,b1,P)
				self.dist[j,i] = (-b0/b1,1.0/b1,b1*b1*P)

	def solve(self,testdata):
		fg = self.g.factorize(self.potential)
		for i in range(testdata.size):
			if i in self.ev:
				fg.V[i].value = testdata[i]
			else:
				fg.V[i].value = None
			
		solver = GaBP(fg).infer()
		pred = deepcopy(testdata)
		for i in range(testdata.size):
			if i in self.ev:continue
			pred[i] = solver.MPE(i)
		return pred

	
	def predict(self,testdata):
		order = self.g.toposort(reverse = True)
		message = {}
		# from leaves to root
		for i in order:
			parents = self.g.find_parents(i)
			children = self.g.find_children(i)

			if len(parents) == 0:
				continue

			j = parents[0]
			if j in self.ev:
				continue

			if i not in self.ev:
				P0 = 0 ; X0 = 0
				for c in children:
					mu,P = message[c,i]
					P0 += P
					X0 += mu*P

				b0,b1,P = self.dist[i,j]
				############ OR we can create N(1,P-> \inf) to approximate######
				if len(children) == 0:
					mu = (1-b0)/b1
					P = b1*b1*P
				else:
					mu0 = X0/P0
					mu = (mu0-b0)/b1
					P = (b1*b1*P*P0)/(P0+P)
				############################
				message[i,j] = (mu,P)
			else:
				b0,b1,P = self.dist[j,i]
				mu = b0+b1*testdata[i]
				message[i,j] = (mu,P)

		# from root to leaves
		order.reverse()
		for i in order:
			parents = self.g.find_parents(i)
			children = self.g.find_children(i)

			for j in children:
				if j in self.ev:
					continue

				if i not in self.ev:
					P0=0 ; X0=0
					if i == self.root:
						mu,P = self.dist[i]
					else:
						mu,P = message[parents[0],i]
					P0 += P
					X0 += mu*P

					for c in children:
						if c==j: continue
						mu,P = message[c,i]
						P0 += P
						X0 += mu*P

					mu0 = X0/P0
					b0,b1,P = self.dist[i,j]
					############################
					mu = (mu0-b0)/b1
					P = (b1*b1*P*P0)/(P0+P)
					############################
					message[i,j] = (mu,P)
				else:
					b0,b1,P = self.dist[j,i]
					mu = b0+b1*testdata[i]
					message[i,j] = (mu,P)

		# MAP estimation
		prediction = deepcopy(testdata)
		for i in range(self.g.N):
			if i not in self.ev:
				nbs = self.g.find_neighbor(i)
				P0 = 0; X0 =0
				for n in nbs:
					mu,P = message[n,i]
					P0 += P
					X0 += mu*P
				if i == self.root:
					mu,P = self.dist[i]
					P0 += P
					X0 += mu*P
				prediction[i] = X0/P0
		return prediction

	@staticmethod
	def chowliu_tree(data):
		N,D = data.shape
		maxN = (D*(D-1))/2
		curN = 0
		g = Graph(digraph=False)
		for i in range(D):
			n = Node('x{}'.format(i))
			g.add_vertice(n)

		allpair = crossprod(range(D),range(D))
		for i,j in allpair:
			curN += 1; progress(curN,maxN,'Calculate mutual info')
			mu = np.mean(data[:,(i,j)],axis=0)
			var = np.cov(data[:,(i,j)],rowvar=False)
			coef = var[0,1] / np.sqrt(var[0,0]*var[1,1])
			# mutual = - np.log(1-coef*coef)
			g.add_edge(i,j,weight=coef)

		g = g.max_spanning_tree()
		g = g.todirect(0)
		return g

	@staticmethod
	def selftest():
		M = 5
		given = [3,4]
		hidden = tuple([list(filter(lambda x:x not in given,range(M)))])

		# generate fake dataset
		N = 10000
		dataset =[]
		for i in range(N):
			X0 = np.random.normal(100,5)
			X1 = np.random.normal(3*X0-20,10)
			X2 = np.random.normal(0.2*X0+30,5)
			X3 = np.random.normal(2*X1+10,5)
			X4 = np.random.normal(X1+7,5)
			dataset.append([X0,X1,X2,X3,X4])

		# automatically learn the tree structure
		data = np.array(dataset,dtype='float')
		g = GBN.chowliu_tree(data)

		gbn = GBN(g,given)
		trainset = data[0:int(0.8*N),:]
		testset =  data[int(0.8*N):,:]
		gbn.fit(trainset)

		pred = deepcopy(testset)
		R,C = pred.shape

		for i in range(R):
			pred[i,hidden] = gbn.solve(testset[i,:])[hidden]

		# calculate deviation percentage
		deviation = 0.0
		for i in range(R):
			p = pred[i,hidden]
			r = testset[i,hidden]
			deviation += np.sum(abs(p-r)/r)
		print(100*deviation/(R* (M-len(given)) ))

class GDBN:
	def __init__(self,g,ev,intercnt='default'):
		assert(isinstance(g,Graph)), "Input should be a Graph instance"
		assert(g.digraph), "Only directed graph allowed"
		self.g = g
		self.ev = ev
		self.sv = list(filter(lambda x:x not in ev ,range(self.g.N)))
		# construct g2
		if intercnt == 'default': intercnt = [(i,i) for i in self.sv]
		self.g2 = g.expand(intercnt,1)
		self.icpd = {}
		self.cpd = {}

	@staticmethod
	def get_cond_gaussian(mu,sigma):
		mu = mu.reshape((mu.size,1))
		m1 = mu[0,:]
		m2 = mu[1:,:]
		s11 = sigma[0,0]
		s12 = sigma[0:1,1:]
		s21 = s12.T
		s22 = sigma[1:,1:]
		b1 = s12.dot(np.linalg.inv(s22))
		b0 = m1 - b1.dot(m2)
		sigma = s11 - b1.dot(s21)
		b0 = float(b0)
		b1 = list(b1.flatten())
		sigma = float(sigma)
		return b0,b1,sigma

	def fit(self,trainset):
		T,D = trainset[0].shape
		assert(D == self.g.N),"Invalid trainset"
		for elem in trainset:
			T,X = elem.shape
			assert(X==D),"Invalid trainset"
		t0data = np.array([list(item[0,:]) for item in trainset])
		N,D = t0data.shape
		assert(N>1)
		if N<10: print("WARN: too little train samples!")
		# learn initial cpd of state variables
		for v in self.sv:
			parents = self.g.find_parents(v)
			idx = tuple([v] + parents)
			mu = np.mean(t0data[:,idx],axis=0)
			sigma = np.cov(t0data[:,idx],rowvar=False)
			pt = Potential(); pt.ids = idx
			if len(parents)>0:
				pt.P = GDBN.get_cond_gaussian(mu,sigma)
			else:
				pt.P = (float(mu),float(sigma))
			self.icpd[v] = pt

		transition = []
		for item in trainset:
			T,D = item.shape
			for t in range(1,T):
				current = list(item[t-1,:]) + list(item[t,:])
				transition.append(current)
		transition = np.array(transition)
		# learn transition cpd of both state and evidence variables
		for v in range(self.g2.N):
			if v < self.g.N: continue
			parents = self.g2.find_parents(v)
			idx = tuple([v] + parents)
			mu = np.mean(transition[:,idx],axis=0)
			sigma = np.cov(transition[:,idx],rowvar=False)
			pt = Potential(); pt.ids = idx
			pt.P = GDBN.get_cond_gaussian(mu,sigma)
			self.cpd[v-self.g.N] = pt

	def construct_factor_graph(self,T):
		fg = Graph(digraph=False)
		for t in range(T):
			for i in range(self.g.N):
				node = EmptyObject()
				node.type = 'RV'
				node.desc = '{}T{}'.format(self.g.V[i],t)
				node.P = 'na'
				fg.add_vertice(node)

		XN = fg.N
		for v in range(XN):
			if v in self.sv:
				pt = self.icpd[v]
			else:
				pt = self.cpd[v%self.g.N]

			if len(pt.P) == 2: # node potential
				fg.V[v].P = pt.P
			else: # cpd, add factor node
				node = EmptyObject()
				node.type = 'FN'
				node.desc = 'FACTOR'
				offset = v-pt.ids[0]
				node.ids = list(map(lambda x:x+offset,pt.ids))
				b0,b1,sigma = pt.P
				b =  [b0,-1] + b1
				node.P = (b,1.0/sigma)
				fid = fg.add_vertice(node)
				for n in node.ids:
					fg.add_edge(fid,n)
		return fg

	def predict(self,testdata,it=0):
		T,D = testdata.shape
		assert(D == self.g.N), "Invalid test data"
		if it<=0: it=int(1.1*D*T)
		fg = self.construct_factor_graph(T)
		edgeset = fg.get_edges()
		# pre-find the neighbors of RV node
		for i in range(fg.N):
			node = fg.V[i]
			if node.type == 'RV': node.nb = fg.find_neighbor(i)
		message = {}
		# initialize all the messages
		for i,j in edgeset: message[i,j] = (1.0,1.0)
		# loppy GaBP
		for xx in range(it):
			progress(xx+1,it,'Message Passing')
			for i,j in edgeset:
				ni,nj = fg.V[i],fg.V[j]
				# no need for passing message to evidence node
				if (nj.type == 'RV') and ((j%D) in self.ev): continue
				############
				if ni.type == 'RV' and nj.type == 'FN':
					if (i%D) in self.ev:
						vi = testdata[i//D,i%D]
						message[i,j] = (vi,1e+8)
					else:
						X0,P0 = (0,0)
						if ni.P != 'na':
							mu,P = ni.P
							X0 += mu*P
							P0 += P
						for n in ni.nb:
							if n==j:continue
							mu,P = message[n,i]
							X0 += mu*P
							P0 += P
						assert(P0!=0),"Leaf node must be EV"
						message[i,j] = (X0/P0,P0)

				elif ni.type == 'FN' and nj.type == 'RV':
					ids = deepcopy(ni.ids)
					b,P = deepcopy(ni.P)
					if j!=ids[0]:
						idx = ids.index(j)
						ids[0],ids[idx] = ids[idx],ids[0]
						b[1],b[idx+1] = b[idx+1],b[1]

					vmu = []
					vP = [P]
					for nid in ids[1:]:
						mu,P = message[nid,i]
						vmu.append(mu)
						vP.append(P)

					# calculate mu
					summation = 0.0
					for s,v in enumerate(vmu):
						summation += v*b[s+2]
					mu = -(summation+b[0])/b[1]

					# calculate P
					product = np.prod(vP)
					numerator = b[1]*b[1]*product
					denominator = product/vP[0]
					for s in range(2,len(b)):
						denominator += b[s]*b[s]*product/vP[s-1]
					P = numerator/denominator
					message[i,j] = (mu,P)

				else:
					assert(False), "Invalid Factor graph!"

		# calculate marginal distribution
		prediction = deepcopy(testdata)
		for i in range(fg.N):
			node = fg.V[i]
			if node.type == 'RV' and (i%D) in self.sv:
				if node.P == 'na':
					X0,P0=0,0
				else:
					mu,P = node.P
					X0 = mu*P
					P0 = P
				for n in node.nb:
					mu,P = message[n,i]
					X0 += mu*P
					P0 += P
				prediction[i//D,i%D] = X0/P0
		return prediction

	@staticmethod
	def selftest():
		# generate fake time series data
		N = 200; split=int(0.9*N)
		dataset = []
		for n in range(N):
			cur = []
			# generate time 0 data
			x0 = np.random.normal(50,10) # 10 is std, not var
			x1 = np.random.normal(7+0.8*x0,5)
			x2 = np.random.normal(x0-3,5)
			x3 = np.random.normal(x1+3,5)
			cur.append([x0,x1,x2,x3])
			# generate more data
			T = np.random.randint(200,500)
			for t in range(T):
				px = cur[-1]
				x0 = np.random.normal(2+0.9*px[0],5)
				x1 = np.random.normal(5+0.4*px[1]+0.6*x0,5)
				x2 = np.random.normal(x0-4,5)
				x3 = np.random.normal(x1+5,5)
				cur.append([x0,x1,x2,x3])
			dataset.append(np.array(cur,dtype=float))

		trainset = dataset[0:split]
		testset = dataset[split:]

		g = Graph(digraph=True)
		for i in range(4):
			g.add_vertice('X{}'.format(i))
		g.add_edge(0,1)
		g.add_edge(0,2)
		g.add_edge(1,3)

		gdbn = GDBN(g,[2,3])
		gdbn.fit(trainset)
		for item in testset:
			ground = item[:,0:2]
			pred = gdbn.predict(item)[:,0:2]
			print('Average diff is {}'.format( np.sum(abs(ground-pred))/ground.size))