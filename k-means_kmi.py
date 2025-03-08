# Lloyds algorithm (k-means) with k-means++ initialization 

import numpy as np
import matplotlib.pyplot as plt
import random

dataset1 = np.loadtxt('dataset1.csv', delimiter=",")
dataset2 = np.loadtxt('dataset2.csv', delimiter=",")

k = 5

# k-means++ 
def k_means_plus_plus(dataset, k):
    centers = []
    # randomly initialize the first center 
    center_1 = random.randint(0, len(dataset)-1)
    centers.append(dataset[center_1])

    # select remaining centers 
    for i in range(1, k):
        distances = []   
        for dp in dataset:
            centers_distances = []
            for center in centers:
                distance = np.sqrt(np.sum((dp-center)**2))**2
                centers_distances.append(distance)
            min_center_distance = min(centers_distances)
            distances.append(min_center_distance)
        distances = np.array(distances)
        probabilities = distances / distances.sum()
        next_center = np.random.choice(len(dataset), p=probabilities)
        centers.append(dataset[next_center])

    centers = np.array(centers)
    return centers

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



def Lloyds_alg(dataset, k):
    # k-means++ initialization of cluster centers     
    initial_cluster_centers = k_means_plus_plus(dataset, k)
    print(f"init: {initial_cluster_centers}")

    converged = False
    prev = None
    prev_centers = initial_cluster_centers
    curr = None
    curr_centers = None
    while not converged:
        if prev is None:
            prev = assign_dp_to_cluster(dataset, prev_centers)
        else:
            prev = curr
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

            

centers, clusters = Lloyds_alg(dataset1, k)
print(f"cluster at center {centers[0]}\n{clusters[tuple(centers[0])]}")