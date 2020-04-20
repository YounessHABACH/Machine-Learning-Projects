# Support vector Machine (Regression)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import and split the data and classes
dataset = pd.read_csv("../../../datasets/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, 2].values


# features scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(np.reshape(Y, (10,1)))

# Fitting Regression modelto the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,Y)

# predicts a new result with polyn reg
y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[8.5]]))))

# Visualisation the regression result
plt.scatter(x=X, y=Y,color='red')
plt.plot(X, regressor.predict(X), color='green')
plt.title('Truth of Bluff / SVR')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()