import numpy as np

def mean_squared_error(y_true, y_pred):
  '''
  Mean Squared Error (MSE)

  Parameters:
    - y_true: Ground truth target values
    - y_pred: Predicted target values

  Returns:
    - score: MSE score
  '''
  return round(np.mean(np.square(y_true - y_pred)), 2)

def mean_absolute_error(y_true, y_pred):
  '''
  Mean Absolute Error (MAE)

  Parameters:
    - y_true: Ground truth target values
    - y_pred: Predicted target values

  Returns:
    - score: MAE score
  '''

  return round(np.mean(np.abs(y_true - y_pred)), 2)

def huber(y_true, y_pred, delta=1.0):
  '''
  Huber loss

  Parameters:
    - y_true: Ground truth target values
    - y_pred: Predicted target values
    - delta: Threshold for the error

  Returns:
    - loss: Huber loss
  '''
  error = y_true - y_pred # error term between the predictions and true value
  abs_error = np.abs(error)
  quadratic = np.minimum(abs_error, delta) # minimum of the absolute error and delta
  linear = abs_error - quadratic # Remaining part of the absolute error if more than delta
  return np.mean(0.5 * quadratic ** 2 + delta * linear) # Huber loss

def epsilon_insensitive(y_true, y_pred, epsilon=0.1):
  '''
  Epsilon-insensitive loss function which is commonly used in Support Vector Regression (SVR).

  It gives us the flexibility to define a margin of tolerance, epsilon, within which no penalty is associated in the training loss function.
  Parameters:
    - y_true: Ground truth target values
    - y_pred: Predicted target values
    - epsilon: Threshold for the error

  Returns:
    - loss: Epsilon-insensitive loss
  '''
  error = y_true - y_pred # error term between the predictions and true value
  abs_error = np.abs(error)
  return np.mean(np.maximum(abs_error - epsilon, 0)) # Epsilon-insensitive loss


def squared_epsilon_insensitive(y_true, y_pred, epsilon=0.1):
  '''
  Squared Epsilon-insensitive loss function which is commonly used in Support Vector Regression (SVR).

  It gives us the flexibility to define a margin of tolerance, epsilon, within which no penalty is associated in the training loss function.
  Parameters:
    - y_true: Ground truth target values
    - y_pred: Predicted target values
    - epsilon: Threshold for the error

  Returns:
    - loss: Squared Epsilon-insensitive loss
  '''
  error = y_true - y_pred # error term between the predictions and true value
  abs_error = np.abs(error)
  return np.mean(np.square(np.maximum(abs_error - epsilon, 0))) # Squared Epsilon-insensitive loss

def r_squared(y_true, y_pred):
  '''
  Coefficient of determination (R^2)

  Parameters:
    - y_true: Ground truth target values
    - y_pred: Predicted target values

  Returns:
    - score: R^2 score
  '''
  y_true_mean = np.mean(y_true)
  ss_total = np.sum((y_true - y_true_mean) ** 2)
  ss_residual = np.sum((y_true - y_pred) ** 2)
  return round(1 - (ss_residual / ss_total), 2)
