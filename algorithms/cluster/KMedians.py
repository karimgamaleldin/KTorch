from core import BaseUnsupervisedEstimator
import numpy as np
import pandas as pd
from typing import Union
from utils import l1_norm_of_residuals, validate_convert_to_numpy

class KMedians(BaseUnsupervisedEstimator):
    def __init__(self, n_clusters=8, *, init='k-medians++', max_iter=300, tol=1e-4):
        super().__init__(l1_norm_of_residuals)
        assert n_clusters > 0, "Number of clusters must be greater than 0"
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """
        Fit the KMedians model
        
        Params:
          - X: Input features (numpy array or pandas DataFrame)
        """
        X = validate_convert_to_numpy(X)
        self.centroids = self._random_init(X) if self.init == 'random' else self._kmedians_plus_plus_init(X)
        for _ in range(self.max_iter):
            prev_centroids = self.centroids.copy()
            labels = self._assign_clusters(X)
            self.centroids = self._update_centroids(X, labels)
            if np.allclose(prev_centroids, self.centroids, atol=self.tol):
                break

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        '''
        Predict the cluster labels for the input features
        
        Params:
          - X: Input features (numpy array or pandas DataFrame)
        '''
        X = validate_convert_to_numpy(X)
        labels = self._assign_clusters(X)
        return labels 

    def score(self, X: Union[np.ndarray, pd.DataFrame]) -> float:
        '''
        Calculate the inertia of the model

        Returns:
            - inertia: Inertia of the model (float)
        '''
        labels = self.predict(X)
        return self.base_metric(X, self.centroids, labels)
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        '''
        Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers.
        
        Params:
          - X: Input features (numpy array or pandas DataFrame)
        '''
        X = validate_convert_to_numpy(X)
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.sum(np.abs(X - self.centroids[i]), axis=1)
        return distances

    def clone(self) -> 'KMedians':
        """
        Clone the KMedians model
        
        Returns:
          - A clone of the KMedians model
        """
        return KMedians(self.n_clusters, init=self.init, max_iter=self.max_iter, tol=self.tol)
    
    def _random_init(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize the centroids randomly
        
        Params:
          - X: Input features (numpy array)
        """
        centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        return centroids

    def _kmedians_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize the centroids using the k-medians++ algorithm.
        
        Params:
        - X: Input features (numpy array)
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        centroids[0] = X[np.random.choice(X.shape[0])]
        
        for i in range(1, self.n_clusters):
            distances = np.min(np.sum(np.abs(X[:, np.newaxis] - centroids[:i]), axis=2), axis=1)
            probs = distances**2 / np.sum(distances**2)
            centroids[i] = X[np.random.choice(X.shape[0], p=probs)]
        
        return centroids

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assign the input features to the closest cluster
        
        Params:
          - X: Input features (numpy array)
        """
        new_labels = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            new_labels[i] = np.argmin(np.sum(np.abs(X[i] - self.centroids), axis=1))
        return new_labels

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update the centroids of the clusters
        
        Params:
          - X: Input features (numpy array)
        """
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            if np.any(labels == i):
                new_centroids[i] = np.median(X[labels == i], axis=0)
        return new_centroids