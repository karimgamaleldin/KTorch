from core.BaseEstimator import BaseEstimator
import numpy as np
from klearn.tree.DecisionTreeRegressor import DecisionTreeRegressor
from metrics.RegressionMetrics import mean_squared_error, friedman_mse, mean_absolute_error, huber, quantile_loss

class GradientBoostingRegressor(BaseEstimator):
  '''
  Gradient Boosting for regression
  '''

  def __init__(self, loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', max_depth = None, min_samples_split: int =2, max_features=None, min_impurity_decrease=0, ccp_alpha=0.0, alpha=0.9, n_iter_no_change=None, validation_fraction=0.1, tol=1e-4):
    self.loss = loss
    self.learning_rate = learning_rate
    self.n_estimators = n_estimators
    self.subsample = subsample
    self.criterion = criterion
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.max_features = max_features
    self.min_impurity_decrease = min_impurity_decrease
    self.ccp_alpha = ccp_alpha
    self.alpha = alpha
    self.n_iter_no_change = n_iter_no_change
    self.validation_fraction = validation_fraction
    self.tol = tol
    self.trees = []
    self.tree_weights = []
    self.train_errors = []
    

  def fit(self, X, y):
    pass 

  def predict(self, X):
    pass 

  def evaluate(self, X, y, metric):
    pass 

  def clone(self):
    return GradientBoostingRegressor(loss=self.loss, learning_rate=self.learning_rate, n_estimators=self.n_estimators, subsample=self.subsample, criterion=self.criterion, max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features, min_impurity_decrease=self.min_impurity_decrease, ccp_alpha=self.ccp_alpha, alpha=self.alpha, n_iter_no_change=self.n_iter_no_change, validation_fraction=self.validation_fraction, tol=self.tol)
