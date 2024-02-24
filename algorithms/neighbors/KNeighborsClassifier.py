from core.BaseEstimator import BaseEstimator
from metrics.ClassificationMetrics import accuracy_score
from metrics.DistanceMetrics import minkowski_distance
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
    
    # Compute the distance between each test point in X and each training point in self.data
    distances = minkowski_distance(X, self.data, axis=1, p=self.p)

    # Get the indices of the k-nearest neighbors
    indices = np.argsort(distances, axis=0)[:self.n_neighbors]

    # Get the labels of the k-nearest neighbors
    idx_labels = self.labels[indices] # obtain the labels of the k-nearest neighbors

    # Predict the class of each test point
    if self.weights == 'uniform':
      y_pred = np.array([np.bincount(idx_labels[i]).argmax() for i in range(idx_labels.shape[0])])
    else:
      y_pred = []
      epsilon = 1e-6 # to avoid division by zero
      for i, (neighbors_indices, dist) in enumerate(zip(indices, distances)):
        # Calculate the inverse of the distances
        weights = 1 / (dist[:self.n_neighbors] + epsilon)

        # Aggregate weights by class
        class_weights = np.zeros(np.max(self.labels) + 1)
        for idx, weight in zip(neighbors_indices, weights):
          class_weights[self.labels[idx]] += weight

        # Predict the class with the highest weight
        y_pred.append(class_weights.argmax())
      y_pred = np.array(y_pred)
    return y_pred
  
  def score(self, X, y):
    pass 