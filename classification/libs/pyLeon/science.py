# coding:utf-8
import numpy as np
from pyLeon import utils
from pdb import set_trace

class Normalizer:
    def fit(self,X):
        N,F = X.shape
        if N <= F:
            utils.user_warn("In normalizer, #sample is less than features")
        mu = np.mean(X, axis = 0).reshape(1,F)
        std = np.std(X, axis = 0).reshape(1,F)
        self.mu = mu 
        self.std = std
        return self

    def norm(self,X):
        N,F = X.shape
        assert(F == self.mu.size)
        X = X - self.mu
        X = X / self.std
        return X