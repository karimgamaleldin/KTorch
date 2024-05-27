from core import BaseEstimator
from utils.RegressionMetrics import r_squared
from utils.DistanceMetrics import euclidean_distance
import numpy as np

class LocallyWeightedRegressor(BaseEstimator):
  '''
  Locally Weighted Regression
  
  A non-parametric regression algorithm that fits a straight line using weights based on the distance of the points from the target point.
  '''
  def __init__(self, tau=0.5):
    '''
    Initialize the Locally Weighted Regression model.
    
    Params:
      - tau: Bandwidth parameter (float)
    '''
    super().__init__('Locally Weighted Regression', 'non-parametric', r_squared)
    self.tau = tau
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
    
    self.data = np.array(X, dtype=np.float32)
    self.labels = np.array(y, dtype=np.float32)
    
    print('Locally Weighted Regression model fitted successfully')

  def predict(self, X):
    '''
    Generate predictions for the input data.
    
    Params:
      - X: Input features (numpy array or pandas DataFrame)
    '''
    assert self.data is not None, 'Fit the model before making predictions'
    X = np.array(X, dtype=np.float32)

    # Calculate the weights
    distance = euclidean_distance(X, self.data[:, np.newaxis], axis=-1, sqrt=False) # Calculate the Euclidean distance between the input data and the training data
    weights = np.exp(- distance / (2 * self.tau ** 2)).T # Calculate the weights using the Gaussian kernel the transpose is for when we get the weight of the loop we get the ith row of the weights matrix

    # Fit the model
    X_bias = np.c_[np.ones(self.data.shape[0]), self.data] # Add a bias term to the training data
    y_pred = np.zeros(X.shape[0]) # Initialize the predictions array

    for i in range(X.shape[0]):
      w = np.diag(weights[i]) # Create a diagonal matrix of the weights, where the ith row corresponds to the ith input data
      theta = np.linalg.pinv(X_bias.T @ w @ X_bias) @ X_bias.T @ w @ self.labels # Calculate the parameters of the model
      x_i = np.concatenate(([1], X[i])) # Add a bias term to the input data
      y_pred[i] = x_i @ theta # Calculate the prediction for the input data
    return y_pred

  def evaluate(self, X, y, metric):
    preds = self.predict(X)
    return metric(y, preds)
  
  def score(self, X, y):
    return self.evaluate(X, y, self.base_metric)
  
  def clone(self):
    return LocallyWeightedRegressor(self.tau)