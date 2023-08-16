# One of approaches from https://nicholasvadivelu.com/2021/05/10/fast-k-means/
# The notebook plays with different steps to get an under the hood understanding of working of code.
# Refer to the above post for many more approaches
# Especially interesting is the sparse matrices for multiplication.

import numpy as np
import matplotlib.pyplot as plt

def update_labels(data, centers):
    """
    data: n x d  (n samples each d dimensional)
    centers: k x d (k cluster centers - each center is of the same size as data sample)
    return labels: n 
    """
    diff = data[:, None, :] - centers[None, :, :] # n x k x d shape from broadcasting
    distance_squared = np.sum(diff ** 2, axis=2) # reduce along 'd' dimension
    labels = np.argmin(distance_squared, axis=1) # find argmin along 'k' dimension
    return labels

def update_centers(data, labels, k):
    """
    data: n x d  (n samples each d dimensional)
    labels: n (values range from [0, k) indicating cluster center assignment for each data point)
    k : number of clusters (needed because bincount will not use the correct dimension of 'k-1' cluster did not have any samples assigned to it)
    returns centers: k x d
    """
    group_counts = np.bincount(labels, minlength=k) # shape n
    group_sums = np.apply_along_axis(lambda w: np.bincount(labels, weights=w, minlength=k), 0, data)
    new_centers = group_sums / group_counts[:, None]
    return new_centers

def k_means(data: np.ndarray, k: int):
    """
    data is of dims n x d (n data points each of size d)
    k is the number of clusters
    """
    # Random center initialization, can be smarter with k-means++
    centers = data[np.random.randint(len(data), size=k)] # k x d

    while True:
        labels = update_labels(data, centers)
        new_centers = update_centers(data, labels, k)
        if np.all(centers == new_centers): # alternatively, can check for "labels" modification 
            break
        centers = new_centers
    
    return centers, labels


def main():
    # n, k, d = 1007, 7, 23
    n, k, d = 111, 3, 2
    data = np.random.rand(n, d)
    cluster_centers, labels = k_means(data, k)
    print (cluster_centers.shape)
    print (labels.shape)
    print (np.min(labels), np.max(labels))
    print (np.bincount(labels, minlength=k))
    if d == 2:
        fig = plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.3)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=range(k), marker="x")
        plt.show()

if __name__ == "__main__":
    main()
