#!/usr/bin/env python
# coding: utf-8

# Knn for diabetes Use case

# In[1]:


import pandas as pd
import numpy as np
from math import sqrt


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score


# In[43]:


dataset = pd.read_csv("diabetes.csv")


# In[4]:


# replace zeros
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for c in zero_not_accepted:
    dataset[c] = dataset[c].replace(0, np.NaN)
    mean = int(dataset[c].mean(skipna=True))
    dataset[c] = dataset[c].replace(np.NaN, mean)


# In[5]:


# split datasets
X = dataset.iloc[:, :8]
Y = dataset.iloc[:, 8]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.2)


# In[12]:


#feature scaling to scal data between 2 mean values
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[7]:


#define the model : Init KNN with the number of n recommended 1
classifier = KNeighborsClassifier(n_neighbors=int(sqrt(len(dataset))), p=2, metric='euclidean')


# In[8]:


# fit model
classifier.fit(X_train, Y_train)


# In[9]:


outcome_pred = classifier.predict(X_test)


# In[13]:


cm = confusion_matrix(Y_test,outcome_pred)
cm


# In[17]:


ac = accuracy_score(Y_test,outcome_pred)
ac


# In[18]:


f1 = f1_score(Y_test,outcome_pred)
f1


# In[19]:


report = classification_report(Y_test, outcome_pred)
print(report)


# In[ ]:




