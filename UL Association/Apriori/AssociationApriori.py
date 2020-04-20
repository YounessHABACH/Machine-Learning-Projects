# Apriori

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Preprocessing 
dataset = pd.read_csv('../../../datasets/Market_Basket_Optimisation.csv', header=None)

# prepare the transactions correctly
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
# training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions=transactions, min_support=0.003, min_confidente=0.2, min_lift=3, min_length=2)

# visualize the result
results = list(rules)