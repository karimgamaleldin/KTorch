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