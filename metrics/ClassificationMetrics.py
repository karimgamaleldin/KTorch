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