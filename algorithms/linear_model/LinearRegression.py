from core.BaseEstimator import BaseEstimator
from metrics.RegressionMetrics import mean_squared_error
import numpy as np

class LinearRegression(BaseEstimator):
  '''
  Linear Regression

  A regular least squares linear regression model that inherits from the BaseEstimator class.
  '''
  def __init__(self, algorithm_name='Linear Regressor', algorithm_type='linear_model', fit_intercept=True):
    super().__init__(algorithm_name, algorithm_type, mean_squared_error)
    self.fit_intercept = fit_intercept
    self.algorithm_params = {}
    self.algorithm_params['fit_intercept'] = fit_intercept
    self.algorithm_params['weights'] = None
    self.algorithm_params['bias'] = None

  def fit(self, X, y):
    '''
    Fit the linear regression models using the least squares approach

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
    self.algorithm_params['weights'] = (np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y)
    self.algorithm_params['bias'] = self.algorithm_params['weights'][-1]
    self.algorithm_params['weights'] = self.algorithm_params['weights'][:-1]
    print('Least Squares Linear Regression model fitted successfully')

  def predict(self, X):
    '''
    Predict the target values for the input features.

    Parameters:
      - X: Input features (numpy array or pandas DataFrame)

    Returns:
      - y: Predicted target values (numpy array)
    '''
    return self.algorithm_params['bias'] + X @ self.algorithm_params['weights']


