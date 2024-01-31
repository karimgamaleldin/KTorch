from core.BaseEstimator import BaseEstimator
import numpy as np

class QuadraticDiscriminantAnalysis(BaseEstimator):
  def __init__(self, algorithm_name, algorithm_type, algorithm_params):
    super().__init__(algorithm_name, algorithm_type, algorithm_params)