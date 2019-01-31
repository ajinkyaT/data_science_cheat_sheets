#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 09:24:20 2017

@author: ajinkya
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix

train=pd.read_csv("train_sum_ori.csv")
train.drop(["cm_key"],axis=1,inplace=True)

le = preprocessing.LabelEncoder()
le.fit(train.Predicted)
train.Predicted=le.transform(train.Predicted)
class_weight = class_weight.compute_class_weight('balanced', np.unique(train.Predicted), train.Predicted)
class_weights=dict(zip(np.unique(train.Predicted),class_weight))

c=train.drop("Predicted",axis=1).columns.tolist()
trainY=train.Predicted
train=train.drop("Predicted",axis=1)

sc = preprocessing.MinMaxScaler()
sc.fit(train)
train=sc.transform(train)




X_train, X_test, y_train, y_test = train_test_split(train, trainY, 
                                                    test_size = 0.25,stratify=trainY,
                                                    random_state=8)


model = CatBoostClassifier(depth=10, iterations=50, learning_rate=1,
                           loss_function="MultiClass",eval_metric='MultiClass',
                           random_seed=8,thread_count	=4,
                           calc_feature_importance=True,class_weights=class_weight.tolist())

model.fit(X_train
          ,y_train
          ,eval_set = (X_test, y_test)
          ,use_best_model = True
         )

feature_importance = model.get_feature_importance(train,trainY)


fea_imp={'features':c,'importance':feature_importance}
fea_imp=pd.DataFrame(fea_imp)

score=accuracy_score(y_test,model.predict(X_test))
print("Accuracy \n")
print(score)

score=confusion_matrix(y_test,model.predict(X_test))
print("Confusion Matrix \n")
print(score)