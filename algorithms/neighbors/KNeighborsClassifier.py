from core.BaseEstimator import BaseEstimator
from metrics.ClassificationMetrics import accuracy_score
from metrics.DistanceMetrics import minkowski_distance, manhattan_distance, euclidean_distance
import numpy as np

class KNeighborsClassifier(BaseEstimator):
  '''
  K Nearest Neighbors Classifier

  A classifier that uses the k-nearest neighbors algorithm to classify the input data points.
  '''

  def __init__(self, n_neighbors=5, weights='uniform',  p=2, metric='minkowski'):
    '''
    Initialize the K Nearest Neighbors Classifier.

    Params:
      - k: Number of neighbors to consider (int)
      - weights: Weight function used in prediction (str) - 'uniform' or 'distance'
      - p: Power parameter for the Minkowski metric (int
      - metric: Distance metric used to compute the nearest neighbors (str) - 'minkowski', 'manhattan', 'euclidean'
    '''
    super().__init__('K Nearest Neighbors Classifier', 'neighbors', accuracy_score)
    self.n_neighbors = n_neighbors
    self.weights = weights
    self.p = p
    self.metric = metric
    self.data = None
    self.labels = None


  def fit(self, X, y):
    '''
    Fit the model to the training data.

    Params:
      - X: Input features (numpy array or pandas DataFrame)
      - y: Target values (numpy array or pandas Series)
    '''
    assert len(X) == len(y), 'The number of input features and target values must be the same'

    self.data = np.array(X, dtype=np.float64) if isinstance(X, np.ndarray) else X.to_numpy().astype(np.float64)
    self.labels = np.array(y, dtype=np.float64) if isinstance(y, np.ndarray) else y.to_numpy().astype(np.float64)
    
    print('K Nearest Neighbors Classifier model fitted successfully')

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

      # Get the indices of the k-nearest neighbors
      idx = np.argsort(distance)[:self.n_neighbors]
      # Get the labels of the k-nearest neighbors
      idx_labels = self.labels[idx]

      # Predict the target values for the input features
      if self.weights == 'uniform':
        y_pred.append(np.bincount(idx_labels.astype(int)).argmax())
      elif self.weights == 'distance':
        epsilon = 1e-12 # to avoid division by zero
        weights = 1 / (distance[idx] + epsilon)
        y_pred.append(np.bincount(idx_labels.astype(int), weights=weights).argmax()) 
    
    return np.array(y_pred)

  def score(self, X, y):
    '''
    Evaluate the accuracy of the model on the input features and target values.

    Params:
      - X: Input features (numpy array or pandas DataFrame)
      - y: Target values (numpy array or pandas Series)
    Returns:
      - score: Accuracy of the model (float)
    '''
    y_pred = self.predict(X)
    return accuracy_score(y, y_pred) 