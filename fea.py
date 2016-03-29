#!/usr/bin/env python
# coding=utf-8

import numpy as np

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
import pandas as pd
import xgboost as xgb
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA 
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel

#from load_data import load_train, load_test 
def remove_constant(train,test):
    # remove constant columns
    remove = []
    for col in train.columns:
        if train[col].std() < 0.05:
            remove.append(col)
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)
def remove_duplicated(train,test):
    # remove duplicated columns
    remove = []
    c = train.columns
    for i in range(len(c)-1):
        v = train[c[i]].values
        for j in range(i+1,len(c)):
            if np.array_equal(v,train[c[j]].values):
                remove.append(c[j])
    print( remove)
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)

#train_data, labels = load_train()
df_train = pd.read_csv('/Users/jackqiu/data/train.csv')
df_test = pd.read_csv('/Users/jackqiu/data/test.csv')

remove_constant(df_train,df_test)
remove_duplicated(df_train,df_test)
#compute slape_radio
y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1).values
radio = float(np.sum(y_train == 0)) / np.sum(y_train==1)

n_folds = 3
clfs = [
        #RandomForestClassifier(n_estimators=200, bootstrap=True, max_features="auto", max_depth=5, criterion='gini'),
        #RandomForestClassifier(n_estimators=200, bootstrap=True, max_features="auto", max_depth=5, criterion='entropy'),
        #RandomForestClassifier(n_estimators=200, bootstrap=True, max_features="sqrt", max_depth=5, criterion='gini'),
        #ExtraTreesClassifier(n_estimators=100, criterion='gini'),
        #ExtraTreesClassifier(n_estimators=100, criterion='entropy'),
        #GradientBoostingClassifier(subsample=0.65, max_depth=5, n_estimators=300),
        #xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.01, nthread=4, subsample=0.95, colsample_bytree=0.85,scale_pos_weight=1,seed=4242),
        #xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85,scale_pos_weight=10,seed=4242),
        xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=400, learning_rate=0.02, nthread=4, subsample=0.95, colsample_bytree=0.85,scale_pos_weight=3,seed=4242),
        #xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=800, learning_rate=0.01, nthread=4, subsample=0.95, colsample_bytree=0.85,scale_pos_weight=1,seed=4242),
        xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=400, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85,scale_pos_weight=10,seed=4242),
        xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=400, learning_rate=0.02, nthread=4, subsample=0.95, colsample_bytree=0.85,scale_pos_weight=5,seed=4242)
]
et = ExtraTreesClassifier(n_estimators=100, criterion='entropy')

datas, labels = X_train,y_train
et = et.fit(datas,labels)
model = SelectFromModel(et,prefit=True)
local_test = True
fea_select = True
train_datas = datas
train_labels = labels
id_test = df_test['ID']
val_datas = df_test.drop(['ID'], axis=1).values
val_lables = np.zeros((val_datas.shape[0],1))
if fea_select:
    train_datas = model.transform(train_datas)
    val_datas = model.transform(val_datas)

if local_test:
    split_points = datas.shape[0]*0.75
    train_datas = np.array(datas[0:split_points])
    train_labels = np.array(labels[0:split_points])
    #将输入数据进行格式转换
    val_datas = np.array(datas[split_points:])
    val_lables = np.array(labels[split_points:])

print(train_datas.shape)
print(train_labels.shape)
print(val_datas.shape)
print(val_lables.shape)

dataset_blend_train = np.zeros((train_datas.shape[0], len(clfs))) #12000*5
dataset_blend_val = np.zeros((val_datas.shape[0], len(clfs))) #

skf = list(StratifiedKFold(train_labels, n_folds))

for i, clf in enumerate(clfs):
    print ((i, clf)) 
    dataset_blend_val_i = np.zeros((val_datas.shape[0], len(skf)))
    for j, (train_idx, val_idx) in enumerate(skf):
        print (("Fold ", j))
        x_train = train_datas[train_idx]  
        y_train = train_labels[train_idx]
        x_val = train_datas[val_idx]
        y_val = train_labels[val_idx]
        clf.fit(x_train, y_train,early_stopping_rounds=40,eval_metric="auc",eval_set=[(x_val, y_val)])
        prob_val = clf.predict_proba(x_val)[:,1]
        dataset_blend_train[val_idx,i] = prob_val
        dataset_blend_val_i[:, j] = clf.predict_proba(val_datas)[:, 1]#对测试数据的所有数据进行预测，得到新的特征
    dataset_blend_val[:, i] = dataset_blend_val_i.mean(1)

print ("blending")
clf = LogisticRegression()
clf.fit(dataset_blend_train, train_labels)
if local_test:
    print('Overall AUC:', roc_auc_score(val_lables, clf.predict_proba(dataset_blend_val)[:,1]))
y_pred= clf.predict_proba(dataset_blend_val)[:,1]
submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
submission.to_csv("submission.csv", index=False)

#np.savetxt(fname='val.csv', X=prob_val, fmt='0.9f')


