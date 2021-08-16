# coding: utf-8
import numpy as np
from copy import deepcopy
from functools import partial
from pyLeon.graph import Node,Graph
from pyLeon import utils
from pdb import set_trace

class BayesianNetwork:
	def __init__(self,g,ev):
		assert(g.digraph), "Only directed graph allowed"

		for i in range(g.N):
			parents = g.find_parents(i)
			assert(len(parents) <=1), "At most one parent is allowed for each node"

		self.graph = g
		self.ev = ev

	@staticmethod
	def chowliu_tree(data):
		'''
		Learn a chowliu tree structure based on give data
		data: S*N numpy array, where S is #samples, N is #RV (Discrete)
		'''
		_,D = data.shape
		marginals = {}
		# compute single r.v. marginals
		totalnum = D + (D*(D-1))/2
		nownum = 0
		for i in range(D):
			nownum += 1; utils.progress(nownum,totalnum,'Learning chowliu tree')
			values, counts = np.unique(data[:,i], return_counts=True)
			marginals[i] = dict(zip(values, counts))

		# compute joint marginal for each pair
		for i,j in utils.halfprod(range(D)):
			nownum += 1; utils.progress(nownum,totalnum,'Learning chowliu tree')
			values, counts = np.unique(data[:,(i,j)], axis=0 ,return_counts=True)
			values = list(map(lambda x:tuple(x),values))
			marginals[i,j] = dict(zip(values, counts))
			allcomb = utils.crossprod(list(marginals[i].keys()),list(marginals[j].keys()))
			for v in allcomb:
				if v not in marginals[i,j]: marginals[i,j][v] = 0

		# normalize all marginals
		for key in marginals:
			dist = marginals[key]
			summation = sum(dist.values())
			for k in dist: dist[k] = (dist[k]+1) / float(summation) # 1- correction

		mutual = {}
		# compute mutual information
		for i,j in utils.halfprod(range(D)):
			mutual[i,j] = 0
			for vi,vj in marginals[i,j]:
				mutual[i,j] += np.log(marginals[i,j][vi,vj] / (marginals[i][vi] * marginals[j][vj])) * marginals[i,j][vi,vj]

		# find the maximum spanning tree
		G = Graph(digraph=False)
		for i in range(D):
			node = Node('N{}'.format(i))
			node.domain = list(marginals[i].keys())
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
		_,D = traindata.shape
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
				allcomb = utils.crossprod(domain,jdomain)
				for v in allcomb:
					if v not in dist: dist[v] = 1 #1-correction
				# normalize
				for vj in jdomain:
					summation = sum(map(lambda vi:dist[vi,vj],domain))
					for vi in domain:
						dist[vi,vj] /= float(summation)
				self.cpt[i,j] = dist
		return self

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

class Clique(Node):
	def __init__(self):
		self.ids = []
		self.desc='|'

	def add_node(self,nid,n):
		self.ids.append(nid)
		self.desc += '{}|'.format(n)

def moralize(g):
	# will lost weight information
	assert(g.digraph)
	ret = deepcopy(g)
	ret.digraph = False
	ret.remove_all_edges()

	for v in range(g.N):
		parents = g.find_parents(v)
		parents.append(v)
		for s,t in utils.halfprod(parents):
			ret.add_edge(s,t)
	return ret

def find_small_clique(X):
	for i,xi in enumerate(X):
		for j,xj in enumerate(X):
			if i!=j and utils.allin(xi.ids,xj.ids):
				return (i,j)
	return None

def get_elim_order(g,alg = 'min-fill',preserve=None):
	assert(not g.digraph)
	assert(alg in ['min-degree','min-fill'])
	
	unmarked = list(range(g.N))
	edges = g.get_edges()
	elim_order = []
	cliques = []
	while len(unmarked) > 0:
		cost = [0 for x in unmarked]
		for i,v in enumerate(unmarked):
			unmarked_neighbor = list(filter(lambda x: (v,x) in edges,unmarked))
			if alg == 'min-degree':
				cost[i] = len(unmarked_neighbor)
			if alg == 'min-fill':
				cost[i] = len(unmarked_neighbor)*(len(unmarked_neighbor)-1)/2
				for s,t in utils.halfprod(unmarked_neighbor):
					if (s,t) in edges:
						cost[i] -= 1
		besti = None
		bestv = None
		if preserve == None:
			besti = cost.index(min(cost))
			bestv = unmarked[besti]
		else:
			tmp = list(zip(unmarked,cost))
			tmp.sort(key = lambda x:x[1])
			for v,_ in tmp:
				children = preserve.find_children(v)
				marked = list(filter(lambda x:x not in unmarked ,list(range(g.N))))
				if utils.allin(children,marked):
					bestv = v
					besti = unmarked.index(bestv)
					break

		elim_order.append(bestv)
		best_neighbor = list(filter(lambda x: (bestv,x) in edges,unmarked))
		for s,t in utils.diffprod(best_neighbor):
			if (s,t) not in edges:
				edges.append( (s,t) )
		best_neighbor.append(bestv)
		cliques.append(best_neighbor)
		unmarked.pop(besti)

	return elim_order,cliques

def get_junction_tree(g, preserve = None):
	assert(not g.digraph)

	order,cliques = get_elim_order(g,preserve = preserve)

	CLIQUE = []
	for i,elem in enumerate(cliques):
		cq = Clique()
		for rid in elem:
			cq.add_node(rid,g.V[rid])
		cq.elim = [order[i]]
		CLIQUE.append(cq)

	while 1:
		pair = find_small_clique(CLIQUE)
		if pair == None:
			break
		i,j = pair
		for eid in CLIQUE[i].elim:
			CLIQUE[j].elim.append(eid)
		CLIQUE.pop(i)

	# find the maximum spanning tree over the clique graph
	newg = Graph(digraph = False)
	for c in CLIQUE:
		newg.add_vertice(c)

	vertices = range(newg.N)
	for (i,j) in utils.halfprod(vertices):
		cinodes = newg.V[i].ids
		cjnodes = newg.V[j].ids
		weight = sum(map(lambda x:x in cinodes,cjnodes))
		if weight>0: newg.add_edge(i,j,weight=weight)

	newg = newg.max_spanning_tree()
	return newg

class Potential:
	def __repr__(self):
		return 'Potential <{}>'.format(self.ids)

class DynamicBayesianNetwork:
	def __init__(self,g,ev,intercnt='default'):
		'''
		intercnt - the connection between two time slice, 'default' means we have 1-to-1 connection between
		all the state variables.
		'''
		assert(g.digraph), "Only directed graph allowed"
		
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

	def logprob(self,sequence):
		T,F = sequence.shape
		assert(F == self.G.N), "Invalid input data"
		
		sum_log_prob = 0.0
		for t in range(T):
			if t == 0:
				cur_slice = sequence[0,:]
				for i in range(F):
					if i in self.EV:
						potential = self.CPT[i]
					else:
						potential = self.ICPT[i]
					ind = tuple(cur_slice[potential.ids])
					sum_log_prob += np.log(potential.P[ind])
			else:
				slice_two = sequence[(t-1):(t+1),:]
				ex_slice = slice_two.flatten()
				for i in range(F):
					potential = self.CPT[i]
					ind = tuple(ex_slice[potential.ids])
					sum_log_prob += np.log(potential.P[ind])
		avg_log_prob = sum_log_prob/float(T)
		return avg_log_prob		
	
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
			if utils.allin(ids,nids):
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
		if len(p1.ids) == 0:
			return p2

		if len(p2.ids) == 0:
			return p1

		if p1.ids == p2.ids:
			newp = self.init_potential(p1.ids)
			newp.P = p1.P * p2.P
			return newp

		if len(p1.ids) >= len(p2.ids):
			pb = p1; ps = p2
		else:
			pb = p2; ps = p1
		assert(utils.allin(ps.ids,pb.ids))

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

			ret[i] = self.condition_out(ret[i],self.EV,E)
	
	def marginalize(self,pt,ids):
		if len(pt.ids) == 0:
			assert(len(ids) ==0)
			return pt

		if pt.ids == ids:
			newp = deepcopy(pt)
			newp.P = newp.P/np.sum(newp.P)
			return newp

		newp = deepcopy(pt)
		sumout = list(filter(lambda v:v not in ids,pt.ids))
		for s in sumout:
			dim = newp.ids.index(s)
			if self.mode == 'max':
				newp.P = np.amax(newp.P,axis = dim)
			else:
				assert(self.mode == 'sum')
				newp.P = np.sum(newp.P,axis = dim)
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

	def condition_out(self,inP,elim,value):
		# condition out variables in [elim] of potential [inP]
		# the value of c is in list (array) [value]
		newids = utils.notin(inP.ids, elim)
		if len(newids) == 0:
			potential = Potential()
			potential.ids = []
		
		elif len(newids) < len(inP.ids):
			potential = self.init_potential(newids)
			for npi,_ in np.ndenumerate(potential.P):
				idx = [-1 for v in inP.ids]
				for si,v in enumerate(inP.ids):
					if v in elim:
						idx[si] = value[v]
					else:
						idx[si] = npi[newids.index(v)]
				idx = tuple(idx)
				potential.P[npi] = inP.P[idx]
		else:
			potential = inP
		
		return potential

	def smooth(self,data,numnodes=4,smooth=True):
		self.mode = 'max'
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
		T1G = moralize(T1G)
		for bkc in appro:
			for s,t in utils.halfprod(bkc):
				T1G.add_edge(s,t)

		self.J1 = get_junction_tree(T1G,preserve=self.G)

		# find come and out node
		self.J1.out = []
		for bkc in appro:
			self.J1.out.append( self.min_clique(self.J1,bkc) )
		self.J1.come = deepcopy(self.J1.out)

		# create junction tree Jt
		T2G = moralize(self.G2)
		for bkc in appro:
			for s,t in utils.halfprod(bkc):
				T2G.add_edge(s,t)

			fbkc = list(map(lambda x:self.M[x],bkc))
			for s,t in utils.halfprod(fbkc):
				T2G.add_edge(s,t)

		self.J2 = get_junction_tree(T2G,preserve = self.G2)

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
			utils.progress(t+1,T, 'Forward')

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
			utils.progress(T-t,T, 'Backward')

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

	def tmp_func(self,G,msg,evi):
		npt = deepcopy(msg)
		elims = list(self.M.values())
		maxidx = max(elims)
		value = np.zeros((maxidx+1,),dtype = int)
		value -= 1
		for k,v in self.M.items():
			value[v] = evi[0,k]

		for k,P in npt.items():
			npt[k] = self.condition_out(P,elims,value)
		message = self.calculate_msg(G,npt)

		prob = 1.0
		for i in G.out:
			pt = self.collect_msg(G,i,npt,message)
			idx = tuple(evi[1,pt.ids])
			prob *= pt.P[idx]
		
		return np.log(prob)
	
	def condLL(self,data,numnodes=4):
		self.mode = 'sum'
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
		T1G = moralize(T1G)
		for bkc in appro:
			for s,t in utils.halfprod(bkc):
				T1G.add_edge(s,t)

		self.J1 = get_junction_tree(T1G,preserve=self.G)

		# find come and out node
		self.J1.out = []
		for bkc in appro:
			self.J1.out.append( self.min_clique(self.J1,bkc) )
		self.J1.come = deepcopy(self.J1.out)

		# create junction tree Jt
		T2G = moralize(self.G2)
		for bkc in appro:
			for s,t in utils.halfprod(bkc):
				T2G.add_edge(s,t)

			fbkc = list(map(lambda x:self.M[x],bkc))
			for s,t in utils.halfprod(fbkc):
				T2G.add_edge(s,t)

		self.J2 = get_junction_tree(T2G,preserve = self.G2)

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
		logprob = 0.0

		for t in range(T):
			# utils.progress(t+1,T, 'Forward')

			fmsg[t] = {}
			evidence = data[t,:]

			if t==0:
				self.init_message(self.J1,fmsg[t])
				self.multiply_CPT(self.J1,evidence,fmsg[t],init=True)
				pre_evi = data[t-1,:]
				evi = np.vstack([pre_evi,evidence])
				logprob += self.tmp_func(self.J1,fmsg[t],evi)
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

				pre_evi = data[t-1,:]
				evi = np.vstack([pre_evi,evidence])
				logprob += self.tmp_func(self.J2,fmsg[t],evi)

				npt = deepcopy(fmsg[t])
				message = self.calculate_msg(self.J2,npt)
				for i in self.J2.out:
					fmsg[t][i] = self.collect_msg(self.J2,i,npt,message)

			if t==(T-1):
				for i,outid in enumerate(self.J2.out):
					inid = self.J2.come[i]
					fmsg[t][outid,-1] = self.get_message(fmsg[t][outid],fmsg[t][inid],timestep = 1)
		
		avg_logprob = logprob/T

		return avg_logprob

	def get_domain(self,nids):
		n0 = self.G2.V[nids[0]]
		D = n0.domain
		for i in nids[1:]:
			node = self.G2.V[i]
			D = utils.crossprod(D,node.domain)
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
			utils.progress(i+1,M,'DBN learning')
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
		return self
