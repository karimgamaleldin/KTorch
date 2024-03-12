from core.BaseEstimator import BaseEstimator
from metrics.ClassificationMetrics import accuracy_score
import numpy as np
from algorithms.tree.DecisionTreeClassifier import DecisionTreeClassifier

class AdaBoostClassifier(BaseEstimator):
  '''
  An AdaBoost classifier

  It fits a classifier on the dataset and then fits copies of the classifer on weighted versions of the same dataset where we increase the weights of the incorrectly classified instances. 
  '''
  def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0):
    super().__init__('AdaBoost', 'Ensemble', accuracy_score)
    self.base_estimator = estimator if estimator is not None else DecisionTreeClassifier(max_depth=1)
    self.n_estimators = n_estimators
    self.learning_rate = learning_rate
    self.estimators = []
    self.estimator_weights = []
    self.estimator_errors = []
    self.sample_weights = []

  def fit(self, X, y):
    pass

  def predict(self, X):
    pass 

  def evaluate(self, X, y, metric):
    pass

  def clone(self):
    return AdaBoostClassifier(estimator=self.base_estimator, n_estimators=self.n_estimators, learning_rate=self.learning_rate)
