#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 21:42:34 2017

@author: ajinkya
"""

import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix

train=pd.read_csv("train_sum.csv")
train.drop(["cm_key"],axis=1,inplace=True)


le = preprocessing.LabelEncoder()
le.fit(train.Predicted)
train.Predicted=le.transform(train.Predicted)


class_weight = class_weight.compute_class_weight('balanced', np.unique(train.Predicted), train.Predicted)
class_weights=dict(zip(np.unique(train.Predicted),class_weight))

class_weight_sample=[6.5,6.5,1.65,4.75]
class_weights_sample=dict(zip(np.unique(train.Predicted),class_weight_sample))


trainY=train.Predicted
sample_weight=train.Predicted.map(class_weights_sample).values
train=train.drop("Predicted",axis=1)

sc = preprocessing.MinMaxScaler()
sc.fit(train)
train=sc.transform(train)


from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

seed=8
np.random.seed(seed)

rf = RandomForestClassifier(random_state=seed,n_estimators=300,
                            n_jobs=-1,oob_score=True,min_samples_leaf=50,max_features=4)


lg=lgb.LGBMClassifier(nthread=4,seed=seed,n_estimators=300,learning_rate=0.01,colsample_bytree=0.7,num_leaves=50,boosting='dart')


xgb = XGBClassifier(max_depth=3, n_estimators=300,silent=True,nthread=4,
                    seed=seed,learning_rate=0.01)

ada=AdaBoostClassifier(n_estimators=300,random_state=seed,learning_rate=0.01)

voting=VotingClassifier(estimators=[('rf',rf),('lg',lg),('xgb',xgb),('ada',ada)],voting='soft',n_jobs=4,
                                    weights=[1,1,1,1])

voting.fit(train, trainY,sample_weight=sample_weight)

score=accuracy_score(trainY,voting.predict(train))
print("Accuracy \n")
print(score)

score=confusion_matrix(trainY,voting.predict(train))
print("Confusion \n")
print(score)

test=pd.read_csv("leader_sum.csv")
test_key=test.cm_key
test.drop(["cm_key"],axis=1,inplace=True)
test=sc.transform(test)

pred = voting.predict_proba(test)
pred_class=voting.predict(test)
pred_max=np.amax(pred, axis=1)
leader=pd.DataFrame({'cm_key' : []})
leader["cm_key"]=test_key
leader["card"]=pred_class
leader["prob"]=pred_max
leader["card"]= leader["card"].astype('int')
leader["card"]=le.inverse_transform(leader["card"])

sub=leader[leader["card"]!="Others"]
sub=sub.sort_values("prob",ascending=False)
#sub_leader=sub.ix[(((sub.card=='Supp') & (sub.prob > 0.399)) | ((sub.card=='Credit') & (sub.prob > 0.373)) |
#                  ((sub.card=='Elite') & (sub.prob > 0.365))),]

sub_leader=sub.ix[((sub.card=='Supp') | (sub.card=='Elite')|(sub.card=='Credit')),]


sub_leader=sub_leader.sort_values("prob",ascending=True)
sub_leader=sub_leader.iloc[(len(sub_leader)-1000):]
sub_leader=sub_leader.sort_values("card",ascending=False)
sub_leader.drop("prob",axis=1,inplace=True)
sub_leader=sub_leader.reset_index(drop=True)
sub_leader.to_csv("DataMfia_IITKGP_order.csv",header=False,index=False)

