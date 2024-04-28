# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:32:25 2024

@author: omkar
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, classification_report


def predict_model(model, x_test):
    return model.predict(x_test)

def evalute_model(y_test ,predictions):
    model_confusion_matrix = confusion_matrix(y_test , predictions)
    model_accuracy_score = accuracy_score(y_test , predictions)
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    print("Confusion Matrix")
    print(model_confusion_matrix)
    print("\n")
    print("Accuracy of Model:", model_accuracy_score*100,'\n')
    print(classification_report(y_test,predictions))
    print('ROC-AUC Curve \n')
    print('***********************')

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()






