#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:04:06 2018

@author: nagasudheerravela
"""
#Importing Necessary Packages
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from operator import itemgetter
from time import time
from scipy.stats import randint as sp_randint




#importing DataFrame
df = pd.read_csv("/Users/nagasudheerravela/Desktop/Github Projetcs/Blood Donation/train.csv")

#Diving Dataset into Outputs and Inputs
X,y = df.iloc[:,2:6],df.iloc[:,6]

# Create the training and test sets
X_train, X_test, y_train, y_test= train_test_split(X, y,
test_size=0.2, random_state=123)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='binary:logistic'
,
n_estimators=10, seed=123)

#Xgboost

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy for Xgboost model : %f" % (accuracy))

#Output: accuracy for Xgboost model : 0.767241

#XGboost model Tuning after optimization by tuning parameters

alg = xgb.XGBClassifier(objective='binary:logistic')

#Randomized Search  search
clf = RandomizedSearchCV(alg,{
        'min_child_weight': [1, 2, 5, 7,10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }, 
                    verbose=1, 
                    scoring='roc_auc')

#Fitiing model
clf.fit(X_train,y_train)

#Remove hash from below to print parameters 
#print(clf.best_params_)

#Creating xgb instance
xg_cl = xgb.XGBClassifier(objective='binary:logistic',params = clf.best_params_,
n_estimators=10, seed=123)

#Fitting model
xg_cl.fit(X_train, y_train)
# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy for xgb model after tuning parameters: %f" % (accuracy))

#accuracy for xgb model after tuning parameters: 0.767241 
#No changes move on to next model


#Decision trees

# Instantiate the classifier: dt_clf_4
dt_clf_4 = DecisionTreeClassifier(max_depth=4)

# Fit the classifier to the training set
dt_clf_4.fit(X_train, y_train)

# Predict the labels of the test set: y_pred_4
y_pred_4 = dt_clf_4.predict(X_test)

# Compute the accuracy of the predictions: accuracy
accuracy = float(np.sum(y_pred_4==y_test))/y_test.shape[0]
print("accuracy for DecisionTree Classifier:", accuracy)

#Output: accuracy for DecisionTree Classifier: 0.7758620689655172


#ExtraTreesClassifier model
clf = ExtraTreesClassifier()
#Fitting the model                              
clf.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = clf.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy for ExtraTreesClassifier: %f" % (accuracy))

#accuracy for ExtraTreesClassifier: 0.706897

#RandomForestClassifier model
clf = RandomForestClassifier()
#Fitting the model                              
clf.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = clf.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy for RandomForestClassifier: %f" % (accuracy))

#accuracy for RandomForestClassifier: 0.786897

#For different runs there is change in values of this model
#I have trained the same dataset in Weka ,I got aroung 96% efficieny in this
#I want to tune different parameters

# In http://scikit-learn.org/0.15/auto_examples/randomized_search.html ,I found
#good documentation for doing it

# build a classifier
clf = RandomForestClassifier(n_estimators=20)


# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 5),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 5),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)

#Got an accuracy 0f 0.786

#use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 4],
              "min_samples_split": [5, 10 ,15,20],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X_train, y_train)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))
report(grid_search.grid_scores_)

#Got an accuracy 0f 0.81

clf = RandomForestClassifier(min_samples_split = 40, 
                             max_leaf_nodes = 15, 
                             n_estimators = 40, 
                             max_depth = 5,
                             min_samples_leaf = 3)
                             
clf.fit(X, y)



# Predict the labels of the test set: preds
preds = clf.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))



#I have changed my parameters every time and finally submitted my model to
#get an Training accuracy of 97% and submitted my results

df_test = pd.read_csv("/Users/nagasudheerravela/Desktop/Github Projetcs/Blood Donation/test3.csv")

#printing Header
print(df_test.head())

#Seperating input
X_fin = df_test.iloc[:,2:6]

#Predictng output
y_fin = clf.predict_proba(X_fin)

#Final submission
df_subm = pd.DataFrame(
    {'': df_test.iloc[:,1],
     'Made Donation in March 2007': y_fin[:,1],
     })

df_subm.to_csv('out.csv', sep=',')






