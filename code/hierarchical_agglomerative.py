import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D

# between examples use euclideean, between clustersd use single and avvergage linkage 

dataset1 = np.loadtxt('dataset1.csv', delimiter=",")
dataset2 = np.loadtxt('dataset2.csv', delimiter=",")

#* DATASET1
hac_single = AgglomerativeClustering(linkage='single', compute_distances=True).fit(dataset1)
hac_average = AgglomerativeClustering(linkage='average', compute_distances=True).fit(dataset1)

single_linkage_matrix = linkage(dataset1, method='single')
average_linkage_matrix = linkage(dataset1, method='average')


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
p = 100

# Single Linkage Dendrogram
dendrogram(single_linkage_matrix, ax=axes[0], no_labels=True, truncate_mode='lastp', p=p)
axes[0].set_title("Single Linkage Dendrogram")
axes[0].set_xlabel("Data Points")
axes[0].set_ylabel("Distance")

# Average Linkage Dendrogram
dendrogram(average_linkage_matrix, ax=axes[1], no_labels=True, truncate_mode='lastp', p=p)
axes[1].set_title("Average Linkage Dendrogram")
axes[1].set_xlabel("Data Points")

plt.tight_layout()
plt.show()

#* DATASET2
hac_single = AgglomerativeClustering(linkage='single', compute_distances=True).fit(dataset2)
hac_average = AgglomerativeClustering(linkage='average', compute_distances=True).fit(dataset2)

single_linkage_matrix = linkage(dataset2, method='single')
average_linkage_matrix = linkage(dataset2, method='average')


#TODO rewrite how this is done since i didnt write it 
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
p = 100

# Single Linkage Dendrogram
dendrogram(single_linkage_matrix, ax=axes[0], no_labels=True, truncate_mode='lastp', p=p)
axes[0].set_title("Single Linkage Dendrogram")
axes[0].set_xlabel("Data Points")
axes[0].set_ylabel("Distance")

# Average Linkage Dendrogram
dendrogram(average_linkage_matrix, ax=axes[1], no_labels=True, truncate_mode='lastp', p=p)
axes[1].set_title("Average Linkage Dendrogram")
axes[1].set_xlabel("Data Points")

plt.tight_layout()
plt.show()
plt.show()



#* dataset1 scatterplots 
hac = AgglomerativeClustering(linkage='single', distance_threshold=0.3, n_clusters=None)
labels = hac.fit_predict(dataset1)

print("Unique cluster labels:", np.unique(labels))


plt.scatter(dataset1[:, 0], dataset1[:, 1], c=labels, cmap='rainbow')
plt.title("Clusters based on distance threshold = 0.3")
plt.show()

hac = AgglomerativeClustering(linkage='average', distance_threshold=3.0, n_clusters=None)
labels = hac.fit_predict(dataset1)

print("Unique cluster labels:", np.unique(labels))


plt.scatter(dataset1[:, 0], dataset1[:, 1], c=labels, cmap='rainbow')
plt.title("Clusters based on distance threshold = 0.3")
plt.show()


#* dataset2 scatterplots 
hac1 = AgglomerativeClustering(linkage='single', distance_threshold=0.9, n_clusters=None)
labels1 = hac1.fit_predict(dataset2)
hac2 = AgglomerativeClustering(linkage='average', distance_threshold=10, n_clusters=None)
labels2 = hac2.fit_predict(dataset2)

# --- Plotting 3D scatter plots ---
fig = plt.figure(figsize=(14, 6))

# Single Linkage 3D Scatter
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(dataset2[:, 0], dataset2[:, 1], dataset2[:, 2], c=labels1, cmap='rainbow')
ax1.set_title("Dataset2 Single Linkage (Threshold=0.9)")

# Average Linkage 3D Scatter
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(dataset2[:, 0], dataset2[:, 1], dataset2[:, 2], c=labels2, cmap='rainbow')
ax2.set_title("Dataset2 Average Linkage (Threshold=10)")

plt.tight_layout()
plt.show()