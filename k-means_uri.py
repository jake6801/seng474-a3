# Lloyds algorithm (k-means) with uniform random initialization 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# load dataset1 into np 2d array
dataset1 = np.loadtxt('dataset1.csv', delimiter=",")
dataset2 = np.loadtxt('dataset2.csv', delimiter=",")

# assign data points to the nearest cluster using euclidean distance     
def assign_dp_to_cluster(dataset, centroids):
    centroid_clusters = {}
    for dp in dataset:
        closest_centroid = None
        min_distance = np.inf 
        for centroid in centroids:
            euclidean_distance = np.sqrt(np.sum((dp-centroid)**2))
            if euclidean_distance < min_distance:
                min_distance = euclidean_distance
                closest_centroid = centroid
        # get the smallest ed and add the dp to that centroid in centroid_clusters 
        if tuple(closest_centroid) not in centroid_clusters:
            centroid_clusters[tuple(closest_centroid)] = [dp]
        else:
            centroid_clusters[tuple(closest_centroid)].append(dp)
    return centroid_clusters

# update centroids 
    # compute new centroid as the mean of all the points in the cluster 
def update_centroids(centroid_clusters):
    new_centroids = []
    for centroid in centroid_clusters:
        new_centroid = np.mean(centroid_clusters[centroid], axis=0)
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

# def check_convergence(dict1, dict2, key_mapping):
#     return all(dict1[key1] == dict2[key2] for key1, key2 in key_mapping.items())

def check_convergence(prev, curr):
    # Convert lists to sets of tuples for comparison
    prev_clusters = {}
    curr_clusters = {}
    for key in prev:
        prev_clusters[key] = sorted(map(tuple, prev[key]))
    for key in curr:
        curr_clusters[key] = sorted(map(tuple, curr[key]))
    return prev_clusters == curr_clusters

def cost(clusters):
    total = 0
    for center, dps in clusters.items():
        center = np.array(center)
        for dp in dps:
            total += np.sum((dp-center)**2)
    return total




# check for convergence 
    # when running algorithm, save old cluster to dp dict and then after each iteration check if the arrays for each centroid contain the same points as the old arrays for each centroid, if so then this is converged, otherwise continue 

def Lloyds_alg(dataset, k):
    # uniform random initialization of cluster centers 
    random_initializations = random.sample(range(len(dataset)), k)
    initial_cluster_centers = dataset[random_initializations]
    print(f"init: {initial_cluster_centers}")

    converged = False
    prev = None
    prev_centers = initial_cluster_centers
    curr = None
    curr_centers = None
    while not converged:
        # prev = assign_dp_to_cluster(dataset, prev_centers) if prev is None else curr
        if prev is None:
            prev = assign_dp_to_cluster(dataset, prev_centers)
        else:
            prev = curr
        # prev_centers = curr_centers if curr_centers is not None else prev_centers
        if curr_centers is not None:
            prev_centers = curr_centers
        else:
            prev_centers = prev_centers
    
        curr_centers = update_centroids(prev)
        print(f"\n\nprev centers:\n{prev_centers}\ncurr centers:\n{curr_centers}")
        curr = assign_dp_to_cluster(dataset, curr_centers)

        converged = check_convergence(prev, curr)
    print("converged")
    return curr_centers, curr

#? initialize k
# k_vals = range(2,16)
k_vals = [4]

#* DATASET1
lowest_costs = []

for k in k_vals:
    lowest_cost = np.inf
    for i in range(5):
        centers, clusters = Lloyds_alg(dataset1, k)
        cluster_cost = cost(clusters)
        if cluster_cost < lowest_cost:
            lowest_cost = cluster_cost
    lowest_costs.append(lowest_cost)

# cost vs k plot 
plt.plot(k_vals, lowest_costs, marker='o')
plt.xlabel("Number of Clusters K")
plt.ylabel("Cost")
plt.show()

# centers, clusters = Lloyds_alg(dataset2, k)
# print(f"cluster at center {centers[0]}\n{clusters[tuple(centers[0])]}")


print("Starting plot")
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

# Assign unique colors to clusters
colors = plt.cm.get_cmap('tab10', len(clusters))  # 'tab10' supports up to 10 colors

# Plot each cluster with a different color
for i, (centroid, points) in enumerate(clusters.items()):
    points = np.array(points)  # Convert list to numpy array
    plt.scatter(points[:, 0], points[:, 1], s=20, color=colors(i), label=f'Cluster {i+1}')

# Plot cluster centers
plt.scatter(centers[:, 0], centers[:, 1], s=200, marker='X', c='red', label='Centroids')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering with Lloyd's Algorithm")
plt.legend()
plt.show()



#* DATASET2
k_vals = [7]
lowest_costs = []

for k in k_vals:
    lowest_cost = np.inf
    for i in range(5):
        centers, clusters = Lloyds_alg(dataset2, k)
        cluster_cost = cost(clusters)
        if cluster_cost < lowest_cost:
            lowest_cost = cluster_cost
    lowest_costs.append(lowest_cost)

# cost vs k plot 
plt.plot(k_vals, lowest_costs, marker='o')
plt.xlabel("Number of Clusters K")
plt.ylabel("Cost")
plt.show()

centers, clusters = Lloyds_alg(dataset2, 15)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink']

for i, (center, dps) in enumerate(clusters.items()):
    dps = np.array(dps)
    colour = colours[i % len(colours)]
    ax.scatter(dps[:, 0], dps[:, 1], dps[:, 2], c=colour, marker='o', alpha=0.6)    
    center = np.array(center)
    ax.scatter(center[0], center[1], center[2], c=colour, marker='X', s=200, edgecolors='k')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D K-Means Clustering Visualization")
plt.show()