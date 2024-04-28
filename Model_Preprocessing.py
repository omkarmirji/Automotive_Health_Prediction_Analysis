# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 18:52:13 2024

@author: omkar
"""

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## EDA --> Exploring the data and figuring out Missing values in the Data

def preprocessing(df):
    print(df.info())
    print(df.isnull().sum())
    print(df.describe())
    print(df['Engine Condition'].value_counts())

def univariate_analysis(df):
    plt.figure(figsize = (20,12))
    list_columns =  df.columns[0:6].tolist()
    for index, col_name in enumerate(list_columns):
        plt.subplot(2,3,index+1)
        sns.kdeplot(x= col_name, hue='Engine Condition', data=df)
    plt.show()

def multivariate_analysis(df):
    plt.figure(figsize = (20,12))
    list_columns =  df.columns[0:6].tolist()
    for index, col_name in enumerate(list_columns):
        plt.subplot(2,3,index+1)
        sns.boxplot(y = col_name , x= 'Engine Condition', hue='Engine Condition', data=df)
    plt.show()

