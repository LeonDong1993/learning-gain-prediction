# coding: utf-8
import sys
sys.path.insert(0,'../data/')
import mlib_data
import numpy as np
from pyLeon import utils
from pyLeon.potential import CLG
from multiprocessing import Pool
from cvxfeature import find_cvx_hull,extract_cvx_feature
from functools import partial
from tqdm import tqdm
from sklearn.decomposition import PCA
from pdb import set_trace
MAX_CORE = None

def preliminary_feature(timearr,mov,flag = 1):
    dmov = mlib_data.take_one_order(mov)
    ddmov = mlib_data.take_one_order(dmov)
    delt = timearr[1:] - timearr[0:-1]
    vec = np.array([item.total_seconds() for item in delt])
    vec = 1.0/vec
    # vec = vec/np.min(vec)
    vec = vec.reshape(-1,1)
    dmov *= vec
    assert(flag == 1), 'ddmov is not implemeted now'
    tmp = {1: dmov, 2:ddmov, 3: np.concatenate((dmov,ddmov),axis = 0)}
    data = tmp[flag]
    set_trace()
    data = data.flatten()
    return data

def gaussian_feature(mov,order = 2):
    N,F = mov.shape
    feature = []
    for f in range(F):
        data = mov[:,f]
        train = np.zeros((N-order,order+1))
        for i in range(order+1):
            train[:,order-i] = data[i:(N-order+i)]
        clg = CLG()
        clg.fit(train)
        A,b,S = clg.para
        feature += list(A.flatten()) + [b,S]
    return feature

def get_cvx_feature(x,listK):
    ret = {}
    inds = find_cvx_hull(x, k = max(listK),verbose = False)
    for k in listK:
        xx = extract_cvx_feature(x,inds[0:k],verbose = False)
        ret[k] = xx
    return ret

def get_pca_feature(x,listK):
    ret = {}
    for k in listK:
        extractor = PCA(n_components = k, svd_solver = 'arpack').fit(x)
        feaX = extractor.transform(x)
        ret[k] = feaX
    return ret

def _run_(args,fix):
    l,s = args
    raw = fix[0]
    feature_para = fix[1]
    fea_func = fix[2]
    ret = {}
    ##### main code below #######
    if l >= s:
        segdata = mlib_data.cut_series(raw,l,s)
        U,M,L,S = zip(*segdata)
        Y = np.array(L)
        T = np.array(U)
        S = np.array(S)

        assert(M[0].shape[1] == 21) # make sure the orientation of hand is removed

        selection = [range(0,3), range(3,7), range(7,10), range(10,14), range(14,17), range(17,21)]

        X = []
        for ids in selection:
            pX = []
            for mov in M:
                # timearr = mov[:,0]
                # m = mov[:,1:]
                # pX.append(preliminary_feature(timearr, m[:,ids]))
                pX.append(mov[:,ids].flatten()) #2020-4-5 14:45:13
            pX = np.array(pX)
            X.append(pX)

        fx = [fea_func(x,feature_para) for x in X]
        for v in feature_para:
            X = np.concatenate([item[v] for item in fx],axis = 1)
            key = (l,s,v)
            ret[key] = (X,Y,T,S)
    return ret

def main(args):
    if len(args) < 3:
        print('pass me 1. method, 2. source file, 3. dest file')
        exit(0)

    method = args[0]
    fname = args[1]
    outname = args[2]

    if method == 'cvx':
        fea_func = get_cvx_feature
    elif method == 'pca':
        fea_func = get_pca_feature
    else:
        assert(0), 'method not supported'

    segment_length = [30,45,60,75,90,105,120]
    shift_size = [5,7,9,11,13]
    feature_para = [10, 20, 30, 40, 50, 60, 70, 80]

    feature_data = {}
    feature_data['info'] = 'Created with parameters:\n Segment Len:{} \n Shift Size:{} \n Fea Para:{} \n Source:{} \n Type:{}'.format(
        segment_length, shift_size, feature_para, fname, method)
    print(feature_data['info'])

    mlib_data.STEPSIZE = 1 #2020-4-5 14:45:35
    raw = mlib_data.load_data(fname, binary = True)
    mlib_data.STEPSIZE = 15 #2020-4-5 14:45:35
    para_set = list(utils.product([segment_length, shift_size]))
    obj = partial(_run_, fix = (raw,feature_para,fea_func))

    p = Pool(MAX_CORE)
    ret = list(tqdm(p.imap_unordered(obj,para_set),total = len(para_set)))
    p.close(); p.join()

    for dictitem in ret:
        for k,v in dictitem.items():
            feature_data[k] = v
    utils.pickle_dump(feature_data,outname)

if __name__ == "__main__":
    main(sys.argv[1:])