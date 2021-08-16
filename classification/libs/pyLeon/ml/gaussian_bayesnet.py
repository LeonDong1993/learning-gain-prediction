import numpy as np
from copy import deepcopy
from pdb import set_trace
from pyLeon.potential import CLG
from pyLeon.graph import Graph,Node
from pyLeon import utils
from pyLeon.solver import GaBP

class GBN:
    def __init__(self,g,ev=None):
        assert(g.digraph), "Only directed graph allowed"
        self.g = g
        self.ev = ev
        self.potential = {}
        self.mass_correction = 1e-100
    
    def fit(self,train_data, weight = None):
        _,D = train_data.shape
        assert(self.g.N == D), "Input data not valid"

        for i in range(self.g.N):
            parents = self.g.find_parents(i)
            domain = [i] + parents
            clg = CLG().fit(train_data[:,domain], weight = weight)
            clg.domain = domain
            self.potential[i] = clg
        return self

    def predict(self,testdata,unknown = []):
        if len(unknown) > 0:
            self.ev = utils.notin(range(self.g.N), unknown)
            
        fg = self.g.factorize(self.potential)
        N,D = testdata.shape
        assert(D == self.g.N), "Number of feature does not match!"
        pred = deepcopy(testdata)

        for r in range(N):
            # progress(r,N,"Predict")

            row = testdata[r,:]
            for i in range(D):
                if i in self.ev:
                    fg.V[i].value = row[i]
                else:
                    fg.V[i].value = None

            solver = GaBP(fg).infer()

            for i in range(D):
                if i not in self.ev:
                    pred[r,i] = solver.MPE(i)
        return pred

    def gradient(self,x,i):
        gd = self.mass(x)
        summation = 0
        pt = self.potential[i]
        summation += pt.gradient(x[pt.domain], 0, logf = True)
        children = self.g.find_children(i)
        for c in children:
            pt = self.potential[c]
            summation += pt.gradient(x[pt.domain], 1, logf = True)
        gd *= summation
        return gd

    def mass(self,x):
        density = 1.0
        for i in range(self.g.N):
            p = self.potential[i]
            dom = p.domain
            density *= p.prob(x[dom])

        if density < self.mass_correction:
            utils.user_warn('In GBN, mass underflow, correction applied')
            density += self.mass_correction
        return density

    @staticmethod
    def chowliu_tree(data,root=0, sample = False):
        _,D = data.shape
        g = Graph(digraph=False)
        for i in range(D):
            n = Node('x{}'.format(i))
            g.add_vertice(n)

        allpair = utils.halfprod(range(D))
        for i,j in allpair:
            coef = CLG.corr_coef(data[:,(i,j)])
            g.add_edge(i,j, weight = coef)

        g = g.max_spanning_tree(sample)
        g = g.todirect(root)
        return g

    @staticmethod
    def selftest():
        M = 5
        given = [1,2]
        hidden = tuple(utils.notin(range(M),given))

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
        cut = int(0.8*N)
        trainset = data[0:cut,:]
        testset =  data[cut:,:]

        g = GBN.chowliu_tree(data)
        gbn = GBN(g,given).fit(trainset)
        pred = gbn.predict(testset)

        # calculate deviation percentage
        unknown = utils.notin(range(M),given)
        ideal = testset[:,unknown]
        result = pred[:,unknown]
        diff = abs(result - ideal)
        pct_diff = np.sum(diff)/np.sum(abs(ideal))
        print('Average diff is {:.2f}%'.format(100*np.mean(pct_diff)))

if __name__ == "__main__":
    GBN.selftest()