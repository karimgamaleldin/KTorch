from core.BaseEstimator import BaseEstimator
from metrics.RegressionMetrics import mean_squared_error, r_squared
from metrics.DistanceMetrics import minkowski_distance, manhattan_distance, euclidean_distance
import numpy as np

class KNeighborsRegressor(BaseEstimator):
  '''
  K Nearest Neighbors Regressor

  A regressor that uses the k-nearest neighbors algorithm to predict the target values for the input data points.
  '''

  def __init__(self, n_neighbors=5, weights='uniform', p=2, metric='minkowski'):
    '''
    Initialize the KNeighborsRegressor
    params:
    - n_neighbors: Number of neighbors to consider (int)
    - weights: Weight function used in prediction (str) - 'uniform' or 'distance'
    - p: Power parameter for the Minkowski metric (int), if p = 1, use manhattan_distance, if p = 2, use euclidean_distance
    - metric: Distance metric used to compute the nearest neighbors (str) - 'minkowski', 'manhattan', 'euclidean'
    '''
    super().__init__('K Nearest Neighbors Regressor', 'neighbors', mean_squared_error)
    self.n_neighbors = n_neighbors
    self.weights = weights
    self.p = p
    self.metric = metric
    self.data = None 
    self.labels = None
  
  def fit(self, X, y):
    '''
    Fit the model to the training data.
    params:
    - X: Input features (numpy array or pandas DataFrame)
    - y: Target values (numpy array or pandas Series)
    '''
    assert len(X) == len(y), 'The number of input features and target values must be the same'

    self.data = np.array(X, dtype=np.float64) if isinstance(X, np.ndarray) else X.to_numpy().astype(np.float64)
    self.labels = np.array(y, dtype=np.float64) if isinstance(y, np.ndarray) else y.to_numpy().astype(np.float64)

    print('K Nearest Neighbors Regressor model fitted successfully') 

  def predict(self, X):
    '''
    Predict the target values for the input features.
    params:
    - X: Input features (numpy array or pandas DataFrame)
    returns:
    - y: Predicted target values (numpy array)
    '''
    # Convert the input features to a numpy array with the correct dtype "np.float64"
    X = np.array(X, dtype=np.float64) if isinstance(X, np.ndarray) else X.to_numpy().astype(np.float64)

    y_pred = []
    for x in X:
      x = x.reshape(1, -1)
      if self.metric == 'minkowski':
        distance = minkowski_distance(x, self.data, p=self.p)
      elif self.metric == 'manhattan':
        distance = manhattan_distance(x, self.data)
      elif self.metric == 'euclidean':
        distance = euclidean_distance(x, self.data)
      
      # Get the index of the k-nearest neighbors
      idx = np.argsort(distance, axis=0)[:self.n_neighbors]
      # Get the labels of the k-nearest neighbors
      idx_labels = self.labels[idx]

      # Predict the target values for the input features
      if self.weights == 'uniform':
        y_pred.append(np.mean(idx_labels, axis=0)) # compute the average of the target values of the k-nearest neighbours
      else :
        eps = 1e-12 # to avoid division by zero
        weights = 1 / (distance[idx] + eps) # compute the weights of each neighbour 
        y_pred.append(np.sum(weights * idx_labels, axis=0) / np.sum(weights, axis=0)) # compute the weighted average of the target values of the k-nearest neighbours

    return np.array(y_pred)



  def score(self, X, y):
    '''
    Compute the coefficient of determination R^2 of the prediction.
    params:
    - X: Input features (numpy array or pandas DataFrame)
    - y: Target values (numpy array or pandas Series)
    returns:
    - score: R^2 of the prediction (float)
    ''' 
    y_pred = self.predict(X)
    return r_squared(y, y_pred) # compute the R^2 score of the prediction

