from core import BaseEstimator
import numpy as np
from utils.ClassificationMetrics import accuracy_score

class Perceptron(BaseEstimator):
  '''
  Perceptron algorithm

  The Perceptron is a simple algorithm suitable for large scale learning, it keeps track of the weights of each feature and updates them when a mistake is made.
  '''
  def __init__(self, *, penalty: str =None, alpha: float =0.0001, l1_ratio: float=0.15, fit_intercept: bool=True, max_iter: int=1000, tol: float=1e-3, eta0: int=1.0, early_stopping: bool=False, n_iter_no_change:int =5):
    '''
    Initialize the Perceptron model with the penalty, regularization strength, and other parameters
    
    params:
    - penalty: str: the penalty term to use for regularization
    - alpha: float: the regularization strength
    - l1_ratio: float: the ratio of L1 regularization to L2 regularization
    - fit_intercept: bool: whether to calculate the intercept for this model
    - max_iter: int: the maximum number of iterations
    - tol: float: the tolerance for the stopping criterion
    - eta0: float: the learning rate for the gradient descent algorithm
    - early_stopping: bool: whether to use early stopping to terminate training when validation score is not improving
    - n_iter_no_change: int: the number of iterations with no improvement to wait before stopping early
    '''
    super().__init__('Perceptron', 'linear_model', accuracy_score)
    assert penalty in [None, 'l2', 'l1', 'elasticnet'], 'penalty must be one of None, l2, l1, elasticnet'
    self.penalty = penalty
    assert alpha > 0, 'alpha must be greater than 0'
    self.alpha = alpha
    assert 0 <= l1_ratio <= 1, 'l1_ratio must be between 0 and 1'
    self.l1_ratio = l1_ratio
    self.fit_intercept = fit_intercept
    assert max_iter > 0, 'max_iter must be greater than 0'
    self.max_iter = max_iter
    assert tol > 0, 'tol must be greater than 0'
    self.tol = tol
    assert eta0 > 0, 'eta0 must be greater than 0'
    self.eta0 = eta0
    self.early_stopping = early_stopping
    assert n_iter_no_change > 0, 'n_iter_no_change must be greater than 0'
    self.n_iter_no_change = n_iter_no_change
    self.weights = None
    self.intercept = None

  def fit(self, X, y):
    '''
    Fit the Perceptron model to the training data
    
    params:
    - X: np.ndarray: the input features
    - y: np.ndarray: the target values
    ''' 
    # Convert the input features and target values to numpy arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # Get the number of samples, features and classes
    n_samples, n_features = X.shape
    classes = np.unique(y)
    assert len(classes) == 2 and set(classes) == {0, 1}, 'Perceptron is a binary classifier, with classes 0 and 1'
    self.weights = np.zeros(n_features)
    self.intercept = 0

    # Fit the model
    prev_loss = np.inf
    iter_no_change = 0
    for iter in range(self.max_iter):

      for i in range(n_samples):
        y_hat = np.dot(X[i], self.weights) + self.intercept # y_hat = w^T * x + b
        if y[i] != (y_hat > 0): # if the prediction is wrong update the weights
            self.weights += self.eta0 * (y[i] - y_hat) * X[i]

            # Apply regularization
            if self.penalty == 'l1':
              self.weights -= self.eta0 * self.alpha * np.sign(self.weights)
            elif self.penalty == 'l2':
              self.weights -= self.eta0 * self.alpha * self.weights
            elif self.penalty == 'elasticnet':
              self.weights -= self.eta0 * self.alpha * (self.l1_ratio * np.sign(self.weights) + (1 - self.l1_ratio) * self.weights)
            
            if self.fit_intercept:
              self.intercept += self.eta0 * (y[i] - y_hat)

      loss = np.sum(np.maximum(0, -y * (np.dot(X, self.weights) + self.intercept))) # Calculate the loss

      # Check if early stopping is enabled
      if self.early_stopping:
        if abs(prev_loss - loss) < self.tol: # Check if the loss is not changing
          iter_no_change += 1
          if iter_no_change >= self.n_iter_no_change:
            break
        else:
          iter_no_change = 0

      prev_loss = loss # Update the previous loss

    print(f'Perceptron trained for {iter + 1} iterations')        

  def predict(self, X):
    '''
    Predict the target values using the trained model

    params:
    - X: np.ndarray: the input features
    ''' 
    out = np.dot(X, self.weights) + self.intercept
    return np.where(out >= 0, 1, 0)

  def evaluate(self, X, y, metric=None):
    preds = self.predict(X)
    return metric(y, preds)
  
  def score(self, X, y):
    return self.evaluate(X, y, self.base_metric)
  
  def clone(self):
    return Perceptron(penalty=self.penalty, alpha=self.alpha, l1_ratio=self.l1_ratio, fit_intercept=self.fit_intercept, max_iter=self.max_iter, tol=self.tol, eta0=self.eta0, early_stopping=self.early_stopping, n_iter_no_change=self.n_iter_no_change)
