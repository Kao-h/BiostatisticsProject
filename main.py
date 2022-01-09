#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 20:36:40 2021

@author: kaouther
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier




# import dataset
features = pd.read_csv("data.csv", header=0, index_col=(0))
# labels
output = pd.read_csv("labels.csv", header=0, index_col=(0))

# data
X = features
Y = output.iloc[:, 0]

#check shape
print(X.shape)
print(Y.shape)



def evaluationFunc(y_test,y_predict):
    #it a function used to evaluate the quality of the model's predictions
    #accuracy 
    print(" the accuracy_score = " , accuracy_score(y_test, y_predict))
    #crosstab
    print(pd.crosstab(y_test, y_predict))

    df_crosstab = pd.crosstab(y_test, y_predict)

    #visulization as a heatmap of the crosstab

    sns.heatmap(df_crosstab, annot=True)
    
    #more detailled quality report 

    print(classification_report(y_test, y_predict))



#Models:

# KNN
#split data 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42,  stratify=Y)

#implement the model
knn = KNeighborsClassifier()

#fit

knn.fit(X_train, y_train)

#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn, param_grid, cv=5)
#fit model to training data
knn_gscv.fit(X, Y)

#check top performing n_neighbors value
print(knn_gscv.best_params_)
# {'n_neighbors': 1}
print(knn_gscv.best_score_)
#0.9987577639751553
plt.plot(knn_gscv.cv_results_['param_n_neighbors'].data, knn_gscv.cv_results_['mean_test_score'])

#predict
y_predict =knn.predict(X_test)
#pred train
y_pred_train =knn.predict(X_train)

# #evaluate

evaluationFunc(y_test, y_predict)



# =====================best k from scartch ===============================
# # search for an optimal value of K for KNN
# # range of k we want to try
# k_range = range(1, 31)
# # empty list to store scores
# k_scores = []
# 
# for k in k_range:
#     # 2. run KNeighborsClassifier with k neighbours
#     knn = KNeighborsClassifier(n_neighbors=k)
#     # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
#     scores = cross_val_score(knn, X, Y, cv=10, scoring='accuracy')
#     # 4. append mean of scores for k neighbors to k_scores list
#     k_scores.append(scores.mean())
# print(k_scores)
# =============================================================================

# Naive Bayes GaussianNB()

#split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

#implement model
gnb = GaussianNB()

#fit the modem 
y_predict = gnb.fit(X_train, y_train).predict(X_test)
#evaluation 
evaluationFunc(y_test, y_predict)

# =============================================================================

# Multi class logistic regression

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42, stratify=Y)

# define the multinomial logistic regression model
model = LogisticRegression(C=1)
# fit the model on the whole dataset
model.fit(X_train, y_train)
#prediction
y_predict = model.fit(X_train, y_train).predict(X_test)
#evaluation 
evaluationFunc(y_test, y_predict)


training_accuracy = []
testing_accuracy = []

index = [1e-10, 1e-5, 1, 1e5, 1e10, 1e20]
for i in index:
    model = LogisticRegression(C=i)
    model.fit(X_train,y_train)
    training_accuracy.append(accuracy_score(y_train, model.predict(X_train)))
    testing_accuracy.append(accuracy_score(y_test, model.predict(X_test)))
    #evaluation
    pd.crosstab(y_test, model.predict(X_test))

print(index)
print(testing_accuracy)
plt.plot(index, training_accuracy, label = 'Training') # plotting t, a separately
plt.plot(index, testing_accuracy, label = 'Testing') # plotting t, b separately
plt.show()

    
# =============================================================================
#SVM

#split data 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42,  stratify=Y)

svc_model = SVC()
#train model

svc_model.fit(X_train, y_train)
#predict
y_predict =  svc_model.predict(X_test)

#evaluation 
evaluationFunc(y_test, y_predict)
    
#=============================================================================
#Random forest
#split data 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42,  stratify=Y)


RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF.fit(X_train, y_train)
y_predict = model.predict(X_test)


evaluationFunc(y_test, y_predict)

#Underfitting in RF
X_lim=X_train.iloc[10:1000,:]
Y_lim=y_train.iloc[10:1000]

RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF.fit(X_lim, Y_lim)
y_predict = model.predict(X_test)


evaluationFunc(y_test, y_predict)
