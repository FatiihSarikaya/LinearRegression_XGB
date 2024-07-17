import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

data = pd.read_csv('../data/boston.csv')
print(data.head())
print(data.columns)
print(data.info())
print(data.shape)

#Checking missing values

print(data.isnull().sum())

#Understanding the correlation between various features in the dataset

correlation = data.corr()
print(correlation)

#constructing a heatmap to understand the correlation

plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.show()

#Splitting the data and Target

X = data.drop(['MEDV'], axis=1)
y = data['MEDV']

print(X)
print(y)

#Splitting the data into Training data and Test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

#Model Training XGBoost Regressor

#loading the model

model = XGBRegressor()

#training the model witn X_train

model_fit = model.fit(X_train, y_train)

#Evaluation
#Prediction on training data

training_data_prediction = model.predict(X_train)
print(training_data_prediction)

#R squared error
score_1 = metrics.r2_score(y_train, training_data_prediction)
print("R scored error: ",score_1)

#Mean Absolute Error

score_2 = metrics.mean_absolute_error(y_train, training_data_prediction)
print("Mean absolute error: ",score_2)




#Prediction on Test Data

test_data_prediction = model.predict(X_test)
print(test_data_prediction)

#R squared error
score_1 = metrics.r2_score(y_test, test_data_prediction)
print("R scored error: ",score_1)

#Mean Absolute Error

score_2 = metrics.mean_absolute_error(y_test, test_data_prediction)
print("Mean absolute error: ",score_2)

#Visualizing the actual Prices and Predicted prices
plt.scatter(y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Acctual Price vs Predicted Price")
plt.scatter(y_test, test_data_prediction)
plt.show()


