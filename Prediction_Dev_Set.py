# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:32:13 2024

@author: omkar
"""

import numpy as np
import pandas as pd
import joblib
from prediction import predict_model,evalute_model

#df=pd.read_csv("C:\OmkaRucha\Local Disk_D\Inttrvu_ai\Automotive_Engine_Health_Prediction\WIP\Automotive_Health_Monitering\engine_data.csv")

#data_dev = df[15627:]

#y_dev = data_dev['Engine Condition']
#x_dev = data_dev.drop(['Engine Condition'],axis = 1)


x_check = np.array([845,4.877239363,3.669304,3.504418,76.30162614,70.4]).reshape(1,-1)

model = joblib.load('model_classifier.pkl')


output = predict_model(model, x_check)
#evalute_model(y_dev ,predictions)
print(output)




