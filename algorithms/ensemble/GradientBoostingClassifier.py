from core.BaseEstimator import BaseEstimator
import numpy as np
from algorithms.tree.DecisionTreeRegressor import DecisionTreeRegressor

class GradientBoostingClassifier(BaseEstimator):
  '''
  Gradient Boosting for classification
  '''
  def __init__(self, learning_rate=0.1, n_estimators=100, max_depth = None, min_samples_split: int =2, max_features=None, min_impurity_decrease=0, ccp_alpha=0.0):
    self.learning_rate = learning_rate
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.max_features = max_features
    self.min_impurity_decrease = min_impurity_decrease
    self.ccp_alpha = ccp_alpha
    self.F0 = None
    self.trees = []


  def fit(self, X, y):
    pass 

  def predict(self, X):
    logit = self.F0
    for tree in self.trees:
      logit += self.learning_rate * tree.predict(X)
    return np.round(1 / (1 + np.exp(-logit))) 

  def evaluate(self, X, y, metric):
    y_pred = self.predict(X)
    return metric(y, y_pred) 

  def clone(self):
    return GradientBoostingClassifier(learning_rate=self.learning_rate, n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features, min_impurity_decrease=self.min_impurity_decrease, ccp_alpha=self.ccp_alpha)