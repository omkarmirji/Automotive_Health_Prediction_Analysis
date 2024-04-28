# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 19:58:51 2024

@author: omkar
"""
# Importing the libraries
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Train/Test Split

def train_test_split_and_features(df,x,y,test_size):
    x_train, x_test, y_train, y_test = train_test_split(x ,y, test_size = test_size, random_state  = 0)
    #print(x.head(5))
    #print(x.columns)
    #features =list(x.columns)
    return x_train, x_test, y_train, y_test

# Fit and Evaluate Model

# 1. Decision Tree Model

from sklearn.tree import DecisionTreeClassifier
def decision_tree_model(max_depth=10, min_samples_split=0.02):
    decision_tree = DecisionTreeClassifier(
        criterion='gini',  # or 'entropy'
        splitter='random',   # or 'random'
        max_depth=max_depth,    # None means unlimited
        min_samples_split=min_samples_split,
        #min_samples_leaf=1,
        #min_weight_fraction_leaf=0.0,
        max_features='sqrt',  # or 'auto', 'sqrt', 'log2'
        random_state=32,
        #max_leaf_nodes=None,
        #min_impurity_decrease=0.0,
        #min_impurity_split=None,
        #class_weight=None,
        #ccp_alpha=0.0
    )

    return decision_tree

# 2. Random Forest Model
from sklearn.ensemble import RandomForestClassifier
def random_forest_model(n_estimators=100, max_depth=7, min_samples_split=0.02, max_samples=0.05):
    random_forest  = RandomForestClassifier(n_estimators=n_estimators, 
                           criterion='gini', 
                           max_depth=max_depth, 
                           min_samples_split=min_samples_split, 
                           max_features='sqrt', 
                           #max_leaf_nodes=None, 
                           verbose=1,  
                           max_samples=max_samples)    
    
    return random_forest

# Fitting the Model

def fit_model(model, x_train, y_train): 
    return model.fit(x_train, y_train)


def feature_importance(model , x):
    importances = pd.DataFrame(model.feature_importances_)
    features = list(x.columns)
    importances['features']  = features
    importances.columns = ['importance', 'feature']
    importances.sort_values(by = 'importance', ascending = True, inplace = True)
    plt.barh(importances.feature,importances.importance)
    return importances.head(10)




