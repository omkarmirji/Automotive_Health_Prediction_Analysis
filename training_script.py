# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 18:55:12 2024

@author: omkar
"""

import numpy as np
import pandas as pd
import joblib
import os 
#os.chdir("C:\OmkaRucha\Local Disk_D\Inttrvu_ai\ML project example-20240427T060302Z-001\ML project example")

### Step1: Data Gathering and Loading 
df=pd.read_csv("C:\OmkaRucha\Local Disk_D\Inttrvu_ai\Automotive_Engine_Health_Prediction\WIP\engine_data.csv")
#df.head(5)

### Importing Model_preprocessing.py to carry out EDA

from Model_Preprocessing import preprocessing,univariate_analysis,multivariate_analysis

preprocessing(df)
univariate_analysis(df)
multivariate_analysis(df)

### Train/Dev split (to prevent Data Leakage)

data_train = df[0:15627]

y = data_train['Engine Condition']
x = data_train.drop(['Engine Condition'],axis = 1)

### Train/Test Split for Model Building

from Model_Bulding import train_test_split_and_features

#x_, x_dev, y_, y_dev = train_test_split_and_features(df,x,y,0.1)

x_train, x_test, y_train, y_test = train_test_split_and_features(df,x,y,0.2)

### Fitting the Model

from Model_Bulding import random_forest_model, decision_tree_model,fit_model,feature_importance

rf_model  = random_forest_model()
dt_model = decision_tree_model()

model = fit_model(rf_model, x_train, y_train)

# Prediction on y_test

from prediction import predict_model,evalute_model

print('Accuracy on Test Set \n')
predictions = predict_model(model, x_test)
evalute_model(y_test ,predictions)

# Feature Importance
feature_importance(model,x)

joblib.dump(model , 'model_classifier.pkl')










