import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# between examples use euclideean, between clustersd use single and avvergage linkage 

dataset1 = np.loadtxt('dataset1.csv', delimiter=",")
dataset2 = np.loadtxt('dataset2.csv', delimiter=",")

#* DATASET1
hac_single = AgglomerativeClustering(linkage='single', compute_distances=True).fit(dataset1)
hac_average = AgglomerativeClustering(linkage='average', compute_distances=True).fit(dataset1)

single_linkage_matrix = linkage(dataset1, method='single')
average_linkage_matrix = linkage(dataset1, method='average')


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
