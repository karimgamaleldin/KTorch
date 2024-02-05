from core.BaseEstimator import BaseEstimator
from metrics.ClassificationMetrics import accuracy_score
from metrics.DistanceMetrics import minkowski_distance
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
    self.data = np.array(X) if isinstance(X, np.ndarray) else X.to_numpy().astype(np.float64)
    self.labels = np.array(y) if isinstance(y, np.ndarray) else y.to_numpy().astype(np.float64)
    print('K Nearest Neighbors Classifier model fitted successfully')

  def predict(self, X):
    '''
    Predict the target values for the input features.

    Parameters:
      - X: Input features (numpy array or pandas DataFrame)

    Returns:
      - y: Predicted target values (numpy array)
    '''
    X = np.array(X) if isinstance(X, np.ndarray) else X.to_numpy().astype(np.float64)
    
    # Compute the distance between each test point in X and each training point in self.data
    distances = minkowski_distance(X, self.data, axis=1, p=self.p)

    # Get the indices of the k-nearest neighbors
    k_indices = np.argsort(distances, axis=1)[:, :self.k] # return the order of the indices of the k-nearest neighbors if the array was ordered

    # Get the labels of the k-nearest neighbors
    k_labels = self.labels[k_indices] # obtain the labels of the k-nearest neighbors

    # Predict the class of each test point
    if self.weights == 'uniform':
      y_pred = np.array([np.bincount(k_labels[i]).argmax() for i in range(k_labels.shape[0])])
    else:
      y_pred = []
      epsilon = 1e-5 # to avoid division by zero
      for i, (neighbors_indices, dist) in enumerate(zip(k_indices, distances)):
        # Calculate the inverse of the distances
        weights = 1 / (dist[:self.k] + epsilon)

        # Aggregate weights by class
        class_weights = np.zeros(np.max(self.labels) + 1)
        for idx, weight in zip(neighbors_indices, weights):
          class_weights[self.labels[idx]] += weight

        # Predict the class with the highest weight
        y_pred.append(class_weights.argmax())
      y_pred = np.array(y_pred)
    return y_pred