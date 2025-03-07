# Lloyds algorithm (k-means) with uniform random initialization 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# load dataset1 into np 2d array
dataset1 = np.loadtxt('dataset1.csv', delimiter=",")

#? initialize k
k = 5

# uniform random initialization of the centroids 
random_initializations = random.sample(range(len(dataset1)), k)
initial_cluster_centers = dataset1[random_initializations]

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
def update_centroids(dataset, centroids):
    new_centroids = []
    for centroid in centroids:
        new_centroid = np.mean(centroids[centroid], axis=0)
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

def check_convergence(dict1, dict2, key_mapping):
    return all(dict1[key1] == dict2[key2] for key1, key2 in key_mapping.items())


# check for convergence 
    # when running algorithm, save old cluster to dp dict and then after each iteration check if the arrays for each centroid contain the same points as the old arrays for each centroid, if so then this is converged, otherwise continue 

def Lloyds_alg(dataset, k):
    # uniform random initialization of cluster centers 
    random_initializations = random.sample(range(len(dataset1)), k)
    initial_cluster_centers = dataset[random_initializations]
    print(f"init: {initial_cluster_centers}")

    # assign datapoints to clusters using cluster centers and euclidean distance 
    # original_centroid_clusters = assign_dp_to_cluster(dataset, initial_cluster_centers)

    # update cluster centers 
    # new_cluster_centers = update_centroids(dataset, original_centroid_clusters)

    converged = False
    prev = None
    prev_centers = initial_cluster_centers
    curr = None
    curr_centers = None
    while not converged:
        if prev == None:
            prev = assign_dp_to_cluster(dataset, prev_centers)            
        else:
            prev = curr
            prev_centers = curr_centers

        curr_centers = update_centroids(dataset, prev)
        print(f"curr: {curr_centers}")
        curr = assign_dp_to_cluster(dataset, curr_centers)

        key_mapping = {}
        for i in range(0, len(prev_centers)):
            key_mapping[tuple(prev_centers[i])] = tuple(curr_centers[i])

        if check_convergence(prev, curr, key_mapping):
            print("converged")
            return True
        else:
            print("Didnt")
            

Lloyds_alg(dataset1, k)
       






#         curr = assign_dp_to_cluster(dataset, new_cluster_centers)
#         #! if all values are the same then converged = True
#         prev_values = []
#         for value in prev.values():
#             prev_values.append()
#         curr_values = []
#         for value in curr.values():
#             curr_values.append(value)
#         if prev_values == curr_values:
#             converged = True
#         #! else, set prev = curr and curr = new one 
#         else:
#             prev = curr
#             new_cluster_centers = update_centroids(dataset, curr)
#             # curr = assign_dp_to_cluster(dataset, new_cluster_centers)
#     return 
    
    




# #test 

