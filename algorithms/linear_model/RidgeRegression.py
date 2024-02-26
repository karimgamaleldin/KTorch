from core.BaseEstimator import BaseEstimator
from metrics.RegressionMetrics import mean_squared_error, r_squared
import numpy as np

class RidgeRegression(BaseEstimator):
  '''
  Linear least squares with L2 regularization.
  ||y - Xw||^2 + alpha * ||w||^2]
  '''
  def __init__(self, alpha=1.0, fit_intercept=True):
    super().__init__(algorithm_name='Ridge Regressor', algorithm_type='linear_model', base_metric=mean_squared_error)
    self.weights = None
    self.bias = None
    self.alpha = alpha
    self.fit_intercept = fit_intercept
    
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
    X_with_bias = np.column_stack((X, np.ones_like(y)))  if self.fit_intercept else X # Add a column of 1s to the input features if self.fit_intercept is True
    self.weights = (np.linalg.pinv(X_with_bias.T @ X_with_bias + self.alpha * np.eye(X_with_bias.shape[1])) @ X_with_bias.T @ y) # Solve the normal equation with L2 regularization to get the Weights
    self.bias = self.weights[-1] if self.fit_intercept else 0  # Get the bias if self.fit_intercept is True
    self.weights = self.weights[:-1] if self.fit_intercept else self.weights  # Remove the bias if self.fit_intercept is True

    print('Ridge Regression model fitted successfully')
  
  def predict(self, X):
    '''
    Predict the target values for the input features.

    Parameters:
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
  
  
