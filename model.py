#!/usr/bin/env python
# coding=utf-8

import numpy as np

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold

from load_data import load_train, load_test 

train_data, labels = load_train()

n_folds = 10
clfs = [RandomForestClassifier(n_estimators=80, bootstrap=True, max_features="auto", max_depth=30, criterion='gini'),
        RandomForestClassifier(n_estimators=80, bootstrap=True, max_features="auto", max_depth=30, criterion='entropy'),
        RandomForestClassifier(n_estimators=80, bootstrap=True, max_features="sqrt", max_depth=30, criterion='gini'),
        ExtraTreesClassifier(n_estimators=50, criterion='gini'),
        ExtraTreesClassifier(n_estimators=60, criterion='entropy'),
        GradientBoostingClassifier(subsample=0.5, max_depth=20, n_estimators=50)]

datas, labels = load_train()

split_points = 12000
train_datas = np.array(datas[0:split_points])
train_labels = np.array(labels[0:split_points])
val_datas = np.array(datas[split_points:])
val_lables = np.array(labels[split_points:])

print train_datas.shape
print train_labels.shape
print val_datas.shape
print val_lables.shape

dataset_blend_train = np.zeros((train_datas.shape[0], len(clfs)))
dataset_blend_val = np.zeros((val_datas.shape[0], len(clfs)))

skf = list(StratifiedKFold(train_labels, n_folds))

for i, clf in enumerate(clfs):
    print i, clf 
    dataset_blend_val_i = np.zeros((val_datas.shape[0], len(skf)))
    for j, (train_idx, val_idx) in enumerate(skf):
        print "Fold ", j
        x_train = train_datas[train_idx]  
        y_train = train_labels[train_idx]
        x_val = train_datas[val_idx]
        y_val = train_labels[val_idx]
        
        clf.fit(x_train, y_train)
        prob_val = clf.predict_proba(x_val)[:,1]
        dataset_blend_train[val_idx,i] = prob_val
        dataset_blend_val_i[:, j] = clf.predict_proba(val_datas)[:, 1]
    dataset_blend_val[:, i] = dataset_blend_val_i.mean(1)

print 
print "blending"
clf = LogisticRegression()
clf.fit(dataset_blend_train, train_labels)

prob_val = clf.predict_proba(dataset_blend_val)


np.savetxt(fname='val.csv', X=prob_val, fmt='0.9f')


