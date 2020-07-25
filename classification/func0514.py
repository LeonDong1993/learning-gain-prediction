# coding: utf-8
import os,sys
import numpy as np
from multiprocessing import Pool
from collections import defaultdict

from pdb import set_trace
from pyLeon import utils
from pyLeon.misc import Logger

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from tune_hyper import gendata,mlib_data, split_into_groups, split_users, sequence_accuracy

split_ratio = 0.8
split_seed = 0
type_weight = 'balanced'




def user_statistics(obj,typeset):
    X,Y_real,T,S = typeset
    uids = np.unique(T)
    Y_pred = obj.decision_function(X)

    seq_acc = []
    for v in uids:
        selector = T == v
        pred = Y_pred[selector]
        real = Y_real[selector]
        
        label = real[0]
        Z = np.sum(abs(pred))
        pred /= Z
        tmp = [-np.sum(pred[pred<0]), np.sum(pred[pred>0])]
        confidence = round(tmp[int(label)],3)
        
        pred[pred>0] = 1
        pred[pred<0] = 0 
        seg_acc = round(np.sum(pred == real)/real.size,3)

        if confidence <= 0.5:
            t = 0 
        else:
            t = 1
    
        if label == 0:
            label = 'Low'
        elif label == 1:
            label = 'High'
        else:
            assert(0)

        seq_acc.append((v,label,seg_acc,t,confidence))
    seq_acc.sort(key=lambda x:x[0])
    
    # neat print
    for item in seq_acc:
        print(','.join(map(str,item)))


def FUNC(fpath,conf):
    lseg, ss, p_fea, C, gamma = conf
    feature_data = utils.pickle_load(fpath)
    target_data = feature_data[lseg,ss,p_fea]
    X,Y,T,S  = target_data
    userinfo = np.unique(np.vstack([T,Y]),axis = 1)
    tmp = list(zip(userinfo[0,:],userinfo[1,:]))
    groups = split_users(tmp,ratio = split_ratio, seed = split_seed )
    train,test = split_into_groups((X,Y,T,S),groups)

    ################
    obj = SVC(class_weight = type_weight, C = C, gamma = gamma, probability = True).fit(train[0],train[1])
    train_acc = obj.score(train[0],train[1])
    test_acc = obj.score(test[0],test[1])
    train_seq_acc = sequence_accuracy(obj,train)
    test_seq_acc = sequence_accuracy(obj,test)
    print(train_acc,test_acc, train_seq_acc,test_seq_acc)

    result = user_statistics(obj,target_data)


def main():
    data_file = 'fdata/cvx9.pkl'
    best_conf = (120, 7, 50, 10000, 0.07)
    FUNC(data_file,best_conf)

if __name__ == '__main__':
    main()