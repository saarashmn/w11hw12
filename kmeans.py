import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#mixture model
num_samples = 1000
num_clusters = 3
cluster_means = np.array([[0, 0], [3, 3], [-3, 3]])
cluster_covs = np.array([[[1, 0], [0, 1]], [[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]]])
cluster_probs = np.array([0.4, 0.3, 0.3])

# data from the mixture model
data = np.zeros((num_samples, 2))
for i in range(num_samples):
    cluster = np.random.choice(num_clusters, p=cluster_probs)
    data[i] = np.random.multivariate_normal(cluster_means[cluster], cluster_covs[cluster])

#  K-means 
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(data)


plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.show()
