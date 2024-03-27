from core.BaseEstimator import BaseEstimator
import numpy as np

class KMeans(BaseEstimator):
  '''
  KMeans Clustering
  
  A class that represents the KMeans clustering algorithm.
  '''

  def __init__(self, n_clusters: int = 8, max_iters: int = 300, tol = 1e-4):
    '''
    Initialize the KMeans class
    Params:
      - n_clusters: The number of clusters to form
      - max_iters: The maximum number of iterations to run the algorithm
      - tol: The tolerance to declare convergence
    '''
    super().__init__('KMeans Clustering', 'cluster', None)
    self.n_clusters = n_clusters
    self.max_iters = max_iters
    self.tol = tol
    self.centroids = None

  def fit(self, X):
    pass 

  def predict(self, X):
    pass 

  def score(self, X):
    pass 

  def clone(self):
    return KMeans(self.n_clusters, self.max_iters, self.tol)

  def _initialize_centroids(self, X):
    pass

  def _assign_clusters(self, X):
    pass

  def _update_centroids(self, X):
    pass

  def _compute_inertia(self, X):
    pass

  def _has_converged(self, old_centroids, new_centroids):
    pass

  def _plot_clusters(self, X):
    pass

  def _plot_inertia(self, X):
    pass

  def _plot_centroids(self):
    pass