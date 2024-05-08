from core.BaseEstimator import BaseEstimator
from utils.RegressionMetrics import mean_squared_error, r_squared
import numpy as np

class LinearRegression(BaseEstimator):
  '''
  Linear Regression

  A regular least squares linear regression model that inherits from the BaseEstimator class.
  '''
  def __init__(self, fit_intercept=True):
    '''
    Initialize the LinearRegression class
    Params:
      - fit_intercept: A boolean indicating whether to fit an intercept term in the model
    '''
    super().__init__('Linear Regressor', 'linear_model', mean_squared_error)
    self.fit_intercept = fit_intercept
    self.weights = None
    self.bias = None

  def fit(self, X, y):
    '''
    Fit the linear regression models using the least squares approach
    Para,s:
      - X: Input features (numpy array or pandas DataFrame)
      - y: Target values (numpy array or pandas Series)

    ''' 

    # Check if the input features and target values are numpy arrays
    if not isinstance(X, np.ndarray):
      X = X.to_numpy().astype(np.float64)
    if not isinstance(y, np.ndarray):
      y = y.to_numpy().astype(np.float64)

    # Calculate the weights and bias using the normal equation
    X_with_bias = np.column_stack((X, np.ones_like(y)))  if self.fit_intercept else X # Add a column of 1s to the input features if self.fit_intercept is True
    self.weights = (np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y)
    self.bias = self.weights[-1] if self.fit_intercept else 0  # Get the bias if self.fit_intercept is True
    self.weights = self.weights[:-1] if self.fit_intercept else self.weights  # Remove the bias from the weights if fit_intercept is True

    print('Least Squares Linear Regression model fitted successfully')

  def predict(self, X):
    '''
    Predict the target values for the input features.
    Params:
      - X: Input features (numpy array or pandas DataFrame)
    Returns:
      - y: Predicted target values (numpy array)
    '''
    return self.bias + X @ self.weights
  
  def score(self, X, y):
    '''
    Calculate the R-squared value of the model
    Params:
      - X: Input features (numpy array or pandas DataFrame)
      - y: Target values (numpy array or pandas Series)
    Returns:
      - r2: R-squared value (float)
    '''
    y_pred = self.predict(X)
    return r_squared(y, y_pred)
  
  def clone(self):
    '''
    Create a copy of the estimator.
    Returns:
      - estimator: A new instance of the estimator
    '''
    return LinearRegression(fit_intercept=self.fit_intercept)

