# Random Forrest regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import and split the data and classes
dataset = pd.read_csv("../../../datasets/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, 2].values

"""" train test split data set from 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
"""
"""
# features scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(np.reshape(Y, (10,1)))
"""

# Fitting Regression modelto the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X,Y)

# predicts a new result with polyn reg
y_pred = regressor.predict([[6.5]])

# Visualisation the regression result
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x=X, y=Y,color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='green')
plt.title('Truth of Bluff / Random forrest regression model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()