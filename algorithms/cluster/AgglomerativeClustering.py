from core import BaseUnsupervisedEstimator
import numpy as np
import pandas as pd
from typing import Union
from utils import inertia, validate_convert_to_numpy, euclidean_distance, manhattan_distance

class AgglomerativeClustering(BaseUnsupervisedEstimator):
    def __init__(self, n_clusters=8, *, metric='euclidean', linkage='single'):
        '''
        Initialize the AgglomerativeClustering model
        
        Params:
          - n_clusters: Number of clusters to form
          - metric: The distance metric to use
          - linkage: The linkage criterion to use
        '''
        super().__init__(inertia)
        assert not (linkage == 'ward' and metric != 'euclidean'), "Ward linkage requires euclidean metric"
        assert n_clusters > 0, "Number of clusters must be greater than 0"
        self.n_clusters = n_clusters
        assert metric in ['euclidean', 'manhattan'], "Invalid metric"
        self.metric = metric
        assert linkage in ['complete', 'average', 'single'], "Invalid linkage"
        self.linkage = linkage 

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        '''
        Fit the AgglomerativeClustering model
        
        Params:
          - X: Input features (numpy array or pandas DataFrame)
        '''
        X = validate_convert_to_numpy(X)

        # Calculate the pairwise distances (distance matrix)
        distances = self._pairwise_distances(X)

        clusters = np.arange(X.shape[0])
        num_clusters = X.shape[0]

        
        # Get the index of the minumum distance
        while num_clusters > self.n_clusters:
            min_x, min_y = np.unravel_index(np.argmin(distances), distances.shape)

            # Merge the clusters
            clusters[clusters == min_y] = min_x

            # Update the distance matrix by replacing the min_y row and column with inf 
            if self.linkage == 'ward':
                # TODO: Implement ward linkage
                pass
            elif self.linkage == 'complete':
                distances[min_x] = np.maximum(distances[min_x], distances[min_y])
                distances[:, min_x] = np.maximum(distances[:, min_x], distances[:, min_y]) 
            elif self.linkage == 'average':
                distances[min_x] = (distances[min_x] + distances[min_y]) / 2
                distances[:, min_x] = (distances[:, min_x] + distances[:, min_y]) / 2
            elif self.linkage == 'single':
                distances[min_x] = np.minimum(distances[min_x], distances[min_y])
                distances[:, min_x] = np.minimum(distances[:, min_x], distances[:, min_y])

            distances[min_y] = np.inf
            distances[:, min_y] = np.inf
            np.fill_diagonal(distances, np.inf)
            num_clusters -= 1

        unique_clusters = np.unique(clusters)
        for i, cluster in enumerate(unique_clusters):
            clusters[clusters == cluster] = i

        self.labels = clusters


    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        '''
        Fit the AgglomerativeClustering model and predict the cluster labels
        
        Params:
          - X: Input features (numpy array or pandas DataFrame)
          
        Returns:
            - labels: Cluster labels (numpy array)
        '''
        self.fit(X)
        return self.labels


    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        '''
        Predict the cluster labels for the input features
        
        Params:
          - X: Input features (numpy array or pandas DataFrame)

        Returns:
            - labels: Cluster labels (numpy array)
        '''
        raise NotImplementedError("The predict method is not applicable on the AgglomerativeClustering.")

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        '''
        Transform X to a cluster-distance space.
        
        Params:
          - X: Input features (numpy array or pandas DataFrame)
          
        Returns:
        - distances: Distances to each cluster center (numpy array)
        '''
        raise NotImplementedError("The transform method is not applicable on the AgglomerativeClustering.")
        

    def score(self, X: Union[np.ndarray, pd.DataFrame]) -> float:
        '''
        Calculate the inertia of the model
        
        Returns:
            - inertia: Inertia of the model (float)
        '''
        raise NotImplementedError("The score method is not applicable on the AgglomerativeClustering.")

    def clone(self) -> 'AgglomerativeClustering':
        '''
        Clone the AgglomerativeClustering model
        
        Returns:
            - model: Cloned AgglomerativeClustering model
        '''
        return AgglomerativeClustering(self.n_clusters, metric=self.metric, linkage=self.linkage)

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        '''
        Calculate the pairwise distances between the input features
        
        Params:
          - X: Input features (numpy array)
          - metric: The distance metric to use
        '''
        if self.metric == 'euclidean':
            distances = euclidean_distance(X[:, None], X, axis=-1)
        elif self.metric == 'manhattan':
            distances = manhattan_distance(X[:, None], X, axis=-1)

        np.fill_diagonal(distances, np.inf)
        return distances
