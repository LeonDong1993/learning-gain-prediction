import numpy as np
import time
from pdb import set_trace
from copy import deepcopy
from pyLeon.utils import progress
from functools import partial
from multiprocessing import Pool

def dist(p,q):
    # evaluate the distance between point p and q in R^m
    assert(p.shape == q.shape)
    return np.linalg.norm(p-q)

def distance_p2ch(q,S, EPS = 5e-2, maxiter = 100, verbose = False):
    # q - numpy row vector, a point in R^m
    # S - mxn numpy array, the set of points in R^m
    # return the distance of q to the convex hull of S approximately
    
    # find the closet point in S to q
    tmp = [dist(p,q) for p in S]
    ind = np.argmin(tmp)
    # init point
    t = S[ind,:] 
    # record the convex combination weight
    weight = np.zeros(S.shape[0])
    weight[ind] = 1.0
    last_diff = 0.0

    for i in range(maxiter):
        # find most extreme point in S in the direction of q-t
        vec = q-t
        tmp = [np.inner(vec,p-t) for p in S]
        ind = np.argmax(tmp)
        # break if no such point
        if tmp[ind] <= 0:
            break
            
        p = S[ind,:]
        vec = p - t
        # project q-t into vec
        Z = np.dot(vec,vec)
        ratio = vec.dot(q-t)/Z
        newt = t + ratio * vec
        weight *= 1-ratio
        weight[ind] += ratio
        
        # break if coverge
        last_diff = dist(newt,t) #np.max(np.abs(newt-t))
        t = newt
        
        if last_diff < EPS:           
            break
    
    if i == (maxiter-1) and verbose:
        print('Hit max iter with diff {}!'.format( last_diff ))
    return dist(q,t),weight
    
def find_cvx_hull(X,k,verbose = False):
    N,F = X.shape
    if N <= F:
        print('WARN: #Samples less than #Feature')
    ###################
    ratio = 0.1
    pct = 0.1
    multiple = 3.0
    ###################
    candidates = np.array(list(range(X.shape[0])))
    idxes = [0]

    init_dist = -1
    last_dist = -1

    t_start = time.time()
    for i in range(k):
        if verbose:
            progress(i+1,k,"Calculating CVX Hull")
        
        B = X[idxes,:]
        D = np.array([distance_p2ch(item,B)[0] for item in X[candidates,:]])
        ind = np.argmax(D)
        idxes.append(candidates[ind])

        maxdis = D[ind]
        thresh = max(ratio*maxdis,np.percentile(D,pct*100))
        selector = D > thresh
        if np.sum(selector) > multiple*(k-i):
            candidates = candidates[selector]      
        
        if i == 0:
            init_dist = maxdis
        else:
            last_dist = maxdis
        
    t_end = time.time()
    if verbose:
        print('Time:{:.2f} secs DeductionRatio:{:.2f}'.format(t_end-t_start, 1-last_dist/init_dist))

    return idxes

def extract_cvx_feature(X,inds,verbose = False):
    B = X[inds,:]
    ret = []
    for i, item in enumerate(X):
        if verbose:
            progress(i+1,X.shape[0],"Extracting Feature")
        ret.append(distance_p2ch(item,B,EPS=1e-2)[1])
    return np.array(ret)

def main():
    import random
    import matplotlib.pyplot as plt
    # generate fake dataset in a unit circle
    N = 100
    points = []
    
    for _ in range(N):
        r = np.sqrt(np.sqrt(random.random()))
        x = (random.random()*2 - 1) * r
        y = np.sqrt(r**2 - x**2)
        if random.random() < 0.5:
            y = -y
        points.append( (x,y) )
    
    points = np.array(points)

    inds = find_cvx_hull(points,20)
    B = points[inds,:]
    C = extract_cvx_feature(points,B)
    print('{} -> {}'.format(points.shape,C.shape))
    
    plt.figure()
    plt.scatter(points[:,0],points[:,1])
    plt.scatter(B[:,0],B[:,1],marker = '*',c = 'r')
    plt.show()
    

if __name__ == "__main__":
    main()
