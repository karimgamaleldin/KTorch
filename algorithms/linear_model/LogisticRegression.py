from core.BaseEstimator import BaseEstimator
from metrics.ClassificationMetrics import accuracy_score
import numpy as np

class LogisticRegression(BaseEstimator):
  '''
  Logistic Regression

  A regular logistic regression model that inherits from the BaseEstimator class.
  '''

  def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, l1_ratio=None):
    super().__init__('Logistic Regressor', 'linear_model', accuracy_score)
    self.fit_intercept = fit_intercept
    self.penalty = penalty
    self.dual = dual
    self.tol = tol
    self.C = C
    self.intercept_scaling = intercept_scaling
    self.class_weight = class_weight
    self.random_state = random_state
    self.solver = solver
    self.max_iter = max_iter
    self.multi_class = multi_class
    self.verbose = verbose
    self.warm_start = warm_start
    self.l1_ratio = l1_ratio

  def fit(self, X, y):
    pass 

  def predict(self, X):
    pass 