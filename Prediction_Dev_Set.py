
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:32:13 2024

@author: omkar
"""

import numpy as np
import pandas as pd
import joblib
from prediction import predict_model,evalute_model

prediction_file_path="C:\Omkar\Work\DS\ML\Automotive_Health_Prediction_Analysis-main\Predictions.csv"

df_predict=pd.read_csv(prediction_file_path)



#y_dev = data_dev['Engine Condition']
x_dev = df_predict.iloc[:, :]

model = joblib.load('model_classifier.pkl')


output = predict_model(model, x_dev)
output_df  = pd.DataFrame({"Engine Condition":output})

predictions_output = pd.concat([df_predict, output_df], axis=1)
predictions_output.to_csv("output.csv")




