# Automotive Health Prediction Analysis
## A. Objective:
The objective of this work is to build a predictive model, for Engine Health condition which predicts whether it is working in a good condition or bad conditon. This decision is based upon various sensor data for Engine. This prediction is carried out with Machine Learning Model, Classification Algorithm.

## B. About the Data
The data is imported from Kaggle source, which contains a csv file 'engine_data.csv'

### Features
The data contains following features
1. Engine rotatione (RPM)
2. Engine Lubricating oil pressure (Bar)
3. Fuel Pressure (Bar)
4. Engine Lubricating oil Temperature (Degree Celcius)
5. Engine Coolant pressure (Bar)
6. Engine Coolant Temperature (Degree Celcius)

### Target Variable
The output variable is Engine Health condition : 0 --> Unhealthy condition, 
                                                 1 --> Healthy condition
### Link to Data
[https://www.kaggle.com/code/arnabk123/engine-health-data](https://www.kaggle.com/datasets/parvmodi/automotive-vehicles-engine-health-dataset)

## C. Installation and Dependancies
1. Python
2. pandas
3. numpy
4. seaborn
5. matplotlib
6. scikit-learn
7. joblib
 
## D. Machine Learning Model used in the project
1. Decison Tree
2. Random Forest Classifier

## E. Results
1. Random Forest Classifier yeilds results with 67% classification accuracy, which means the model predicts 67 times correct of 100 data points.
2. Decision Tree Classifier yeilds 66% accuracy
3. Engine RPM feature holds most dominance effect on Engine health condition   

## F. Conclusion
1. It is evident that, classificaton Machine Learning model can contribute to predict Engine condition status by acquiring sensor data.
2. This can bring down, significant maintainance cost, at the same time safety to passangers
3. The data needs more features such as number of years in service, piston ring wear, belt condition, oil replacement cycle etc which will help to improve model robustness
