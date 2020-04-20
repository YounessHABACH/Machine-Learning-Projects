# Polynomial Linear regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import and split the data and classes
dataset = pd.read_csv("../../../datasets/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, 2].values

"""
# train test split data set from 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linear_regressor1 = LinearRegression()
linear_regressor1.fit(X, Y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree=4)
X_poly = polynomial_regressor.fit_transform(X)
linear_regressor2 = LinearRegression()
linear_regressor2.fit(X_poly, Y)

# Visualize the linear Regression results
plt.scatter(x=X, y=Y,color='red')
plt.plot(X, linear_regressor1.predict(X), color='green')
plt.title('Truth of Bluff / Linear regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualisation the polynomial regression result
plt.scatter(x=X, y=Y,color='red')
plt.plot(X, linear_regressor2.predict(polynomial_regressor.fit_transform(X)), color='green')
plt.title('Truth of Bluff / Linear regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# predicts a new result with linear reg
y_pred1 = linear_regressor1.predict([[6.5]])

# predicts a new result with polyn reg
y_pred2 = linear_regressor2.predict(polynomial_regressor.fit_transform([[6.5]]))