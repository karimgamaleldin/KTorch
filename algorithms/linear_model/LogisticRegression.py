from core.BaseEstimator import BaseEstimator
from metrics.ClassificationMetrics import accuracy_score
from metrics.Functions import sigmoid, softmax
import numpy as np

class LogisticRegression(BaseEstimator):
  '''
  Logistic Regression

  A regular logistic regression model that inherits from the BaseEstimator class.
  '''

  def __init__(self, penalty='l2', tol=1e-4, C=1.0, fit_intercept=True, class_weight=None, max_iter=100, l1_ratio=None):
    '''
    Initialize the logistic regression model with the penalty, tolerance, regularization strength, and other parameters
    params:
    - penalty: str: the penalty term to use for regularization
    - tol: float: the tolerance for the stopping criterion
    - C: float: the regularization strength
    - fit_intercept: bool: whether to fit an intercept term
    - class_weight: dict: the class weights
    - max_iter: int: the maximum number of iterations
    - l1_ratio: float: the ratio of L1 regularization to L2 regularization
    '''
    super().__init__('Logistic Regressor', 'linear_model', accuracy_score)
    self.fit_intercept = fit_intercept
    self.penalty = penalty
    assert self.penalty in ['l1', 'l2', 'elasticnet', 'none'], "penalty must be one of 'l1', 'l2', 'elasticnet', 'none'"
    self.tol = tol
    self.C = C
    assert self.C > 0, "C must be greater than 0"
    self.fit_intercept = fit_intercept
    self.class_weight = class_weight
    self.max_iter = max_iter
    self.l1_ratio = l1_ratio # only used when penalty is 'elasticnet'

  def fit(self, X, y):
    # Get the number of samples, features and classes
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y)) 

    # Initialize the weights
    self.w = np.zeros((n_features, n_classes))
    self.b = np.zeros(n_classes)

    # Fit the model


  def predict(self, X):
    proba = self.predict_proba(X)
    return np.argmax(proba, axis=1) 

  def predict_proba(self, X):
    dot_product = X @ self.w + self.b
    return softmax(dot_product)

  def score(self, X, y):
    pass

  def clone(self):
    return LogisticRegression(self.penalty, self.tol, self.C, self.fit_intercept, self.class_weight, self.max_iter, self.l1_ratio)
  