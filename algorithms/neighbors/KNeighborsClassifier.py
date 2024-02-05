from core.BaseEstimator import BaseEstimator
from metrics.ClassificationMetrics import accuracy_score
import numpy as np

class KNeighborsClassifier(BaseEstimator):
  '''
  K Nearest Neighbors Classifier

  A classifier that uses the k-nearest neighbors algorithm to classify the input data points.
  '''

  def __init__(self, algorithm_name='K Nearest Neighbors Classifier', algorithm_type='neighbors', k=5, weights='uniform', p=2):
    '''
    Initialize the K Nearest Neighbors Classifier.

    Parameters:
      - algorithm_name: Name of the algorithm (str)
      - algorithm_type: Type of the algorithm (str)
      - k: Number of neighbors to consider (int)
      - weights: Weight function used in prediction (str) - 'uniform' or 'distance'
      - p: Power parameter for the Minkowski metric (int), if p = 1, use manhattan_distance, if p = 2, use euclidean_distance
    '''
    super().__init__(algorithm_name, algorithm_type, accuracy_score)
    self.k = k
    self.weights = weights
    self.p = p
    self.data = None
    self.labels = None

  def fit(self, X, y):
    '''
    Fit the model to the training data.

    Parameters:
      - X: Input features (numpy array or pandas DataFrame)
      - y: Target values (numpy array or pandas Series)
    '''
    if not isinstance(X, np.ndarray):
      self.data = X.to_numpy().astype(np.float64)
    if not isinstance(y, np.ndarray):
      self.labels = y.to_numpy().astype(np.float64)
    print('K Nearest Neighbors Classifier model fitted successfully')

  def predict(self, X):
    '''
    Predict the target values for the input features.

    Parameters:
      - X: Input features (numpy array or pandas DataFrame)

    Returns:
      - y: Predicted target values (numpy array)
    '''
    if not isinstance(X, np.ndarray):
      X = X.to_numpy().astype(np.float64)
    
    # Compute the distance between each test point in X and each training point in self.data
    distances = self.MinkowskiDistance(X, self.data)

    # Get the indices of the k-nearest neighbors
    k_indices = np.argsort(distances, axis=1)[:, :self.k] # return the order of the indices of the k-nearest neighbors if the array was ordered

    # Get the labels of the k-nearest neighbors
    k_labels = self.labels[k_indices] # obtain the labels of the k-nearest neighbors

    # Predict the class of each test point
    if self.weights == 'uniform':
      y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=k_labels)
    else:
      y_pred = np.apply_along_axis(lambda x: np.bincount(x, weights=1 / distances[0]).argmax(), axis=1, arr=k_labels)
    return y_pred

  def MinkowskiDistance(self, X1, X2):
    '''
    Compute the Minkowski distance between two sets of points.

    Parameters:
      - X1: First set of points (numpy array)
      - X2: Second set of points (numpy array)

    Returns:
      - distance: Minkowski distance between the two sets of points (numpy array)
    '''
    return np.sum(np.abs(X1 - X2) ** self.p, axis=1) ** (1 / self.p)
  


