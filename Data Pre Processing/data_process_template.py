# data pre processing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import and split the data and classes
dataset = pd.read_csv("../../datasets/Data.csv")
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, 3]

"""
# preprocess take care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])
"""

# train test split data set from 
from sklearn.model_selection import train_test_split
X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)


"""
# features scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
Y_train = sc_X.transform(X_test)
"""

