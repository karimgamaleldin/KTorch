from core.BaseEstimator import BaseEstimator
from metrics.RegressionMetrics import mean_squared_error
import numpy as np

class RidgeRegression(BaseEstimator):
  '''
  Linear least squares with L2 regularization.
  ||y - Xw||^2_2 + alpha * ||w||^2_2
  '''
  def __init__(self,  algorithm_name='Ridge Regressor', algorithm_type='linear_model', alpha=1.0):
    super().__init__(algorithm_name=algorithm_name, algorithm_type=algorithm_type)
    self.algorithm_params = {}
    self.algorithm_params['weights'] = None
    self.algorithm_params['bias'] = None
    self.algorithm_params['alpha'] = alpha

  def fit(self, X, y):
    '''
    Fit the linear regression models using the normal equation for ridge regression

    Parameters:
      - X: Input features (numpy array or pandas DataFrame)
      - y: Target values (numpy array or pandas Series)

    ''' 

    # Check if the input features and target values are numpy arrays
    if not isinstance(X, np.ndarray):
      X = X.to_numpy().astype(np.float64)
    if not isinstance(y, np.ndarray):
      y = y.to_numpy().astype(np.float64)

    # Calculate the weights and bias using the normal equation
    X_with_bias = np.column_stack((X, np.ones_like(y)))  # Add a bias term (constant) to X
    self.algorithm_params['weights'] = (np.linalg.inv(X_with_bias.T @ X_with_bias + self.algorithm_params['alpha'] * np.eye(X_with_bias.shape[1])) @ X_with_bias.T @ y) # Add the regularization term
    self.algorithm_params['bias'] = self.algorithm_params['weights'][-1]
    self.algorithm_params['weights'] = self.algorithm_params['weights'][:-1]
    print('Ridge Regression model fitted successfully')
  
  def predict(self, X):
    '''
    Predict the target values for the input features.

    Parameters:
      - X: Input features (numpy array or pandas DataFrame)

    Returns:
      - y: Predicted target values (numpy array)
    '''
    return self.algorithm_params['bias'] + X @ self.algorithm_params['weights']
  
  
