import numpy as np

def inertia(X, centroids, labels):
    """
    Calculate the inertia of the clusters
    Params:
      - X: Input features (numpy array)
      - centroids: Cluster centroids (numpy array)
      - labels: Cluster labels (numpy array)
    Returns:
      - inertia: Inertia of the clusters (float)
    """
    inertia_sum = 0
    for i in range(len(centroids)):
        inertia_sum += np.sum((X[labels == i] - centroids[i]) ** 2)
    return inertia_sum

def l1_norm_of_residuals(X, centroids, labels):
    """
    Calculate the L1 norm of residuals of the clusters
    Params:
      - X: Input features (numpy array)
      - centroids: Cluster centroids (numpy array)
      - labels: Cluster labels (numpy array)
    Returns:
      - L1_norm: L1 norm of residuals of the clusters (float)
    """
    L1_norm = 0
    for i in range(len(centroids)):
        L1_norm += np.sum(np.abs(X[labels == i] - centroids[i]))
    return L1_norm

def silehouette_score(X, labels):
    """
    Calculate the silhouette score of the clusters
    Params:
      - X: Input features (numpy array)
      - labels: Cluster labels (numpy array)
    Returns:
      - silhouette_score: Silhouette score of the clusters (float)
    """
    pass

def davies_bouldin_score(X, labels):
    """
    Calculate the Davies-Bouldin score of the clusters
    Params:
      - X: Input features (numpy array)
      - labels: Cluster labels (numpy array)
    Returns:
      - davies_bouldin_score: Davies-Bouldin score of the clusters (float)
    """
    pass

def dunns_index(X, labels):
    """
    Calculate the Dunn's index of the clusters
    Params:
      - X: Input features (numpy array)
      - labels: Cluster labels (numpy array)
    Returns:
      - dunns_index: Dunn's index of the clusters (float)
    """
    pass