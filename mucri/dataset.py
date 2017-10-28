'''
Created on 2017-2-19

@author: Jun Chen
'''
import numpy as np
from sklearn import cross_validation
from sys import stderr
import os
import pandas as pd
from os.path import isfile
from sklearn.preprocessing.data import normalize


np.random.seed(114973)
           

class RealWorldFeed(object):
    
    
    def generate_data(self, folds=5, shuffle=True):
        Xtrain = [[] for _ in xrange(folds)]
        Xtest  = [[] for _ in xrange(folds)]
        for u in xrange(self.U):
            if len(self.X[u]) >= 10:
                kf = cross_validation.KFold(len(self.X[u]), folds, shuffle=shuffle)
                for i, (train, test) in enumerate(kf):
                    Xtrain[i].extend(self.X[u][train])
                    Xtest[i].extend(self.X[u][test])
        return Xtrain, Xtest
    

class INRIAFeed(RealWorldFeed):
    
    
    def __init__(self, path='../data/inria', shuffle=True):
        super(INRIAFeed, self).__init__()
        self.X = []
        uid = 0
        max_itemid = 0
        for f in os.listdir(path):
            if not isfile(os.path.join(path, f)):
                continue
            values = pd.read_csv(os.path.join(path, f), sep='\t', usecols=[2,3], dtype=int, header=None).values
            max_itemid = np.max([max_itemid, np.max(values)])
            values = np.append(values, np.zeros((values.shape[0], 1), dtype=int) + uid, axis=1)
            uid += 1
            if shuffle:
                np.random.shuffle(values)
            self.X.append(values)
        self.U = uid
        self.I = max_itemid + 1
        print 'INRIA: %d users, %d items, %d comparisons' % (self.U, self.I, sum([len(comps) for comps in self.X]))
    
    
class AestheticFeed(RealWorldFeed):
    
    
    def __init__(self, path='../data/aesthetic', shuffle=True):
        super(AestheticFeed, self).__init__()
        self.X = []
        uid = 0
        max_itemid = 0
        for f in os.listdir(path):
            values = pd.read_csv(os.path.join(path, f), sep=',', dtype=int, header=None).values
            max_itemid = max(max_itemid, values.max())
            values = np.append(values, np.zeros((values.shape[0], 1), dtype=int) + uid, axis=1)
            uid += 1
            if shuffle:
                np.random.shuffle(values)
            self.X.append(values)
        self.U = uid
        self.I = max_itemid + 1
        print 'Aesthetic: %d users, %d items, %d comparisons' % (self.U, self.I, sum([len(comps) for comps in self.X]))

 

if __name__ == '__main__':
    
    feed = INRIAFeed()
    feed.generate_data()
    
    feed = AestheticFeed()
    feed.generate_data()
    
    
    