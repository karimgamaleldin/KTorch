class BaseEstimator:
  '''
  Base class for all algorithms for example: linear regression, logistic regression, decision tree, random forest, etc.
  '''
  def __init__(self, algorithm_name, algorithm_type, algorithm_params):
    self.algorithm_name = algorithm_name
    self.algorithm_type = algorithm_type
    self.algorithm_params = algorithm_params
  
  def fit(self, X, y):
    '''
    Fit the model to the training data.

    Parameters:
      - X: Input features (numpy array or pandas DataFrame)
      - y: Target values (numpy array or pandas Series)
    '''
    raise NotImplementedError('Fit method must be implemented in the subclass')
  
  def predict(self, X):
    '''
    Predict the target values for the input features.

    Parameters:
      - X: Input features (numpy array or pandas DataFrame)

    Returns:
      - y: Predicted target values (numpy array)
    '''
    raise NotImplementedError('Predict method must be implemented in the subclass')
  
  def evaluate(self, X, y, metric):
    '''
    Evaluate the model on the given test data.

    Parameters:
      - X: Input features (numpy array or pandas DataFrame)
      - y: Target values (numpy array or pandas Series)
      - metric: Evaluation metric (function)

    Returns:
      - score: Evaluation score (float)
    '''
    raise NotImplementedError('Evaluate method must be implemented in the subclass')
  
  def get_parameters(self):
    '''
    Get the parameters of the model.

    Returns:
      - parameters: Dictionary of model parameters
    '''
    return self.algorithm_params
  
  def get_algorithm_details(self):
    '''
    Get the details of the model.

    Returns:
      - details: Dictionary of model details
    '''
    return {
      'name': self.algorithm_name,
      'type': self.algorithm_type,
      'parameters': self.algorithm_params
    }
