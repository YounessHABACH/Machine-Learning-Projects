# data pre processing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import and split the data and classes
dataset = pd.read_csv("../../../datasets/Salary_Data.csv")
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, 1]


# train test split data set from 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3)

# fit the simple linear regression to training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# predict the test set result
y_pred = regressor.predict(X_test)

#visualization the training set
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'green')
plt.title("Salary vs years of experience - Train")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#visualization the test set
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'green')
plt.title("Salary vs years of experience - Test")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
