import numpy as np

def accuracy_score(y_true, y_pred):
  '''
  Compute the accuracy score.

  Parameters:
    - y_true: True target values (numpy array or pandas Series)
    - y_pred: Predicted target values (numpy array or pandas Series)

  Returns:
    - score: Accuracy score (float)
  '''
  return np.mean(y_true == y_pred)


def binary_cross_entropy(y_true, y_pred):
  '''
  Compute the binary cross-entropy loss.

  Parameters:
    - y_true: True target values
    - y_pred: Predicted target values

  Returns:
    - loss: Binary cross-entropy loss (float)
  '''
  return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_true, y_pred):
  '''
  Compute the categorical cross-entropy loss.

  Parameters:
    - y_true: True target values
    - y_pred: Predicted target values

  Returns:
    - loss: Categorical cross-entropy loss (float)
  '''
  return -np.sum(y_true * np.log(y_pred)) / len(y_true)


def gini_index(y):
  '''
  Compute the Gini index.
  Params:
    - y: target values
  Returns:
    - gini index
  '''
  return 1 - np.sum((np.bincount(y) / len(y))**2)

def entropy(y):
  '''
  Compute the entropy.
  Params:
    - y: target values
  Returns:
    - entropy
  '''
  p = np.bincount(y) / len(y)
  return -np.sum(p * np.log2(p))