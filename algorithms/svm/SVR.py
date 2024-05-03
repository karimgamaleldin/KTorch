from core.BaseEstimator import BaseEstimator
from metrics import mean_squared_error
import numpy as np

class SVR(BaseEstimator):
  '''
  Support Vector Regression
  '''
  def __init__(self, C=1.0, epsilon=0.1, kernel='rbf', gamma='auto', max_iter=1000, coef0=0.0):
    super().__init__('Support Vector Regression', 'svm', mean_squared_error)
    self.C = C
    self.epsilon = epsilon
    self.kernel = kernel
    self.gamma = gamma
    self.max_iter = max_iter
    self.coef0 = coef0

  def fit(self, X, y):
    pass 

  def predict(self, X):
    pass

  def evaluate(self, X, y, metric=mean_squared_error):
    preds = self.predict(X)
    return metric(y, preds)

  def score(self, X, y):
    return self.evaluate(X, y, self.base_metric)
  
  def clone(self):
    return SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel, gamma=self.gamma, max_iter=self.max_iter, coef0=self.coef0)