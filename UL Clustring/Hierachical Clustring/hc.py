# k-means clustring

#importing librairies
import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd

# import data set
dataset = pd.read_csv('../../../datasets/Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the Dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendorgam = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidien distance')
plt.show()
    
# Fitting HC to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X) 

# visualising the clusters
plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='green', label='Cluster 2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='blue', label='Cluster 3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='magenta', label='Cluster 5')
plt.title('Clusters of clients')
plt.xlabel('Annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()

plt.scatter(X[:, 0], X[:, 1])
plt.show()
