from core.BaseEstimator import BaseEstimator
from metrics.RegressionMetrics import mean_squared_error
import numpy as np

class KNeighborsRegressor(BaseEstimator):
  '''
  K Nearest Neighbors Regressor

  A regressor that uses the k-nearest neighbors algorithm to predict the target values for the input data points.
  '''