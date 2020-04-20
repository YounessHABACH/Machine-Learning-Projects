# Artificial Neural Network

"""      Part1 : Data preprocessing      """
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('../../../datasets/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, 13].values
#Encode categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# encode the first feature (Geography)
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# encode the first feature (Sexe)
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#dummy variables for distigue between the non-numerucal vars and numerical
oneHotEncoder = OneHotEncoder(categorical_features=[1])
X = oneHotEncoder.fit_transform(X).toarray()
X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""      Part2 : Making ANN     """
# import keras librairies
import keras
from keras.models import Sequential
from keras.layers import Dense
# initialize the ANN
classifier = Sequential()
# Adding the IL and first HL
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
# Adding the second HL
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
# Adding the OL
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
# compile th ANN and Apply Stochastic Gradient Descent
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit the ANN with the dataset
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

"""      Part3 : Making the prediction     """
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
