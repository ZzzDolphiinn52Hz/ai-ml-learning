import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(18)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]

N = 500

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

# X.shape = (N * 3, 2) = (1500, 2)
X = np.concatenate((X0, X1, X2), axis=0)

K = 3

#--------------- check dataset --------------#
#--------------------------------------------#
print(X0.shape)
print(X1.shape)
print(X2.shape)
print(X.shape) # X.shape = (1500, 2) --> X.shape[0] = 1500, X.shape[1] = 2

# X[i, 0] = x1, X[i, 1] = x2
# column 0 = x1, column 1 = x2
# s = size of the points
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
#--------------------------------------------#
#--------------------------------------------#

# randomly pick k rows from X as initial centroids
def kmeans_init_centroids(X, K):
    return X[np.random.choice(X.shape[0], K, replace=False)]

centroids = kmeans_init_centroids(X, K)
print(centroids)
print(centroids.shape) # --> (3, 2)

#--------------------------------------------#
def kmeans_assign_labels(X, centroids):
    # calculate pairwise distances btw data and centroids follow L2 standard
    D = cdist(X, centroids)
    # return index of the closest centroid
    return np.argmin(D, axis=1) # --> array([1, 0, 2, ..., 1, 0, 2]), point i closest to centroid j

labels = kmeans_assign_labels(X, centroids)
print(labels.shape) # --> (1500,)
print(labels[:20]) # --> 20 first label
#--------------------------------------------#

#--------------------------------------------#
def kmeans_update_centroids(X, labels, K):
    centroids = np.zeros((K, X.shape[1]))
    # empty array to store new centroids
    """
    centroids =
    [
        [0, 0],
        [0, 0],
        [0, 0]
    ]
    """
    for k in range(K):
        Xk = X[labels == k, :]
        centroids[k, :] = np.mean(Xk, axis=0)

    return centroids
#--------------------------------------------#

#--------------------------------------------#
def has_converged(centroids, new_centroids):
    return (
        set([tuple(a) for a in centroids])
        ==
        set([tuple(a) for a in new_centroids])
    )
#--------------------------------------------#

#--------------------------------------------#
def kmeans(X, K):
    centroids = [kmeans_init_centroids(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centroids[-1]))
        new_centroids = kmeans_update_centroids(X, labels[-1], K)
        if has_converged(centroids[-1], new_centroids):
            break
        centroids.append(new_centroids)
        it += 1
    return (centroids, labels, it)
#--------------------------------------------#

#--------------------------------------------#
def kmeans_display(X, labels):
    for k in np.unique(labels):
        plt.scatter(X[labels == k, 0], X[labels == k, 1], s=10)
    final_centroids = centroids[-1]
    plt.scatter(
        final_centroids[:, 0],
        final_centroids[:, 1],
        marker="o",
        s=150
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
#--------------------------------------------#
(centroids, labels, it) = kmeans(X, K)
print("Number of iterations:", it)
print("Centers found by our algorithm:\n", centroids[-1])
kmeans_display(X, labels[-1])