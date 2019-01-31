#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 09:18:56 2017

@author: ajinkya
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

train=pd.read_csv("final_train.csv")
train.drop(["cm_key","mvar1","Predicted"],axis=1,inplace=True)
test=pd.read_csv("leaderboard_final.csv")


test_key=test.cm_key
test.drop(["cm_key","mvar1"],axis=1,inplace=True)


le = preprocessing.LabelEncoder()
le.fit(train.Actual)
train.Actual=le.transform(train.Actual)
train.Actual.value_counts()
class_weight = class_weight.compute_class_weight('balanced', np.unique(train.Actual), train.Actual)
class_weights=dict(zip(np.unique(train.Actual),class_weight))

cat_cols=[9]
trainX=train.drop("Actual",axis=1)
trainY=train.Actual
X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, 
                                                    test_size = 0.4,stratify=train.Actual,
                                                    random_state=8)
model = CatBoostClassifier(depth=10, iterations=10, learning_rate=0.1,
                           loss_function="MultiClass",eval_metric='MultiClass',
                           random_seed=8,thread_count	=4,class_weights=class_weight.tolist())

model.fit(X_train
          ,y_train
          ,cat_features=cat_cols
          ,eval_set = (X_test, y_test)
          ,use_best_model = True
         )

pred = model.predict_proba(test)
pred_class=model.predict(test)
pred_max=np.amax(pred, axis=1)
leader=pd.DataFrame({'cm_key' : []})
leader["cm_key"]=test_key
leader["card"]=pred_class
leader["prob"]=pred_max
leader["card"]= leader["card"].astype('int')
leader["card"]=le.inverse_transform(leader["card"])

sub=leader[leader["card"]!="Others"]
sub=sub.sort_values("prob",ascending=False)
sub_leader=sub.iloc[:1000,]
sub_leader.drop("prob",axis=1,inplace=True)
sub_leader=sub_leader.reset_index(drop=True)
sub_leader.to_csv("leader_board.csv",header=False,index=False)