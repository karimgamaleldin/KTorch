from core.BaseEstimator import BaseEstimator
from metrics.RegressionMetrics import mean_squared_error
import numpy as np

class PoissonRegressor(BaseEstimator):
  '''
  Poisson Regression
  
  A Poisson regression model that inherits from the BaseEstimator class.
  '''

  def __init__(self, alpha: int=1.0, fit_intercept:bool =True, max_iter:int=1000, tol:int=1e-4):
    '''
    Initialize the PoissonRegressor class
    Params:
      - fit_intercept: A boolean indicating whether to fit an intercept term in the model
    '''
    super().__init__('Poisson Regressor', 'linear_model', mean_squared_error)
    self.fit_intercept = fit_intercept
    self.alpha = alpha
    assert self.alpha > 0, "alpha must be greater than 0"
    self.max_iter = max_iter
    assert self.max_iter > 0, "max_iter must be greater than 0"
    self.tol = tol
    assert self.tol > 0, "tol must be greater than 0"
    self.weights = None
    self.bias = None

  def fit(self, X, y):
    pass 

  def predict(self, X):
    pass 

  def evaluate(self, X, y, metric):
    pass 

  def clone(self):
    return PoissonRegressor(self.alpha, self.fit_intercept, self.max_iter, self.tol)
