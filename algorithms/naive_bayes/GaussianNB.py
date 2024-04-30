from core.BaseEstimator import BaseEstimator
import numpy as np
from metrics.ClassificationMetrics import accuracy_score

class GaussianNB(BaseEstimator):
  '''
  Gaussian Naive Bayes
  
  An implementation of Gaussian Naive Bayes algorithm.
  '''
  def __init__(self, prior=None, var_smoothing=1e-9):
    super().__init__('Gaussian Naive Bayes', 'naive_bayes', accuracy_score)
    self.prior = prior
    self.var_smoothing = var_smoothing

  def fit(self, X, y):
    pass 

  def predict(self, X):
    pass

  def evaluate(self, X, y, metric=accuracy_score):
    preds = self.predict(X)
    return metric(y, preds)
  
  def score(self, X, y):
    return self.evaluate(X, y, self._base_metric)
  
  def clone(self):
    return GaussianNB(prior=self.prior, var_smoothing=self.var_smoothing)
  


