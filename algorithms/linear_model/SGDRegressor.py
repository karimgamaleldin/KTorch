from core.BaseEstimator import BaseEstimator
from metrics.RegressionMetrics import mean_squared_error, huber, epsilon_insensitive, squared_epsilon_insensitive
import numpy as np

class SGDRegressor(BaseEstimator):
  '''
  A regularized linear model with stochastic gradient descent (SGD) learning.
  '''

  def __init__(self, loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False):
    super().__init__('SGD Regressor', 'linear_model', mean_squared_error)
    '''
    Initialize the SGDRegressor model.

    Parameters:
      loss (str): The loss function to be used. It should be either "squared_loss", "huber", "epsilon_insensitive", or "squared_epsilon_insensitive".
      penalty (str): The penalty to be used. It should be either "l1", "l2", "elasticnet", or "none".
      alpha (float): The regularization strength.
      l1_ratio (float): The Elastic Net mixing parameter.
      fit_intercept (bool): Whether to fit the intercept.
      max_iter (int): The maximum number of iterations.
      tol (float): The stopping criterion.
      shuffle (bool): Whether to shuffle the data.
      verbose (int): The verbosity level.
      epsilon (float): The epsilon value for the epsilon-insensitive loss function.
      random_state (int): The random state.
      learning_rate (str): The learning rate schedule. It should be either "constant", "optimal", "invscaling", or "adaptive".
      eta0 (float): The initial learning rate for the "constant" or "invscaling" schedules.
      power_t (float): The exponent for the "invscaling" learning rate.
      early_stopping (bool): Whether to use early stopping.
      validation_fraction (float): The proportion of the training data to set aside as validation set for early stopping.
      n_iter_no_change (int): The number of iterations with no improvement to wait before stopping.
      warm_start (bool): Whether to reuse the solution of the previous call to fit as initialization.
      average (bool): Whether to compute the averaged SGD weights.
    '''
    self.loss = loss 
    self.penalty = penalty
    self.alpha = alpha
    self.l1_ratio = l1_ratio
    self.fit_intercept = fit_intercept
    self.max_iter = max_iter
    self.tol = tol
    self.shuffle = shuffle
    self.verbose = verbose
    self.epsilon = epsilon
    self.random_state = random_state
    self.learning_rate = learning_rate
    self.eta0 = eta0
    self.power_t = power_t
    self.early_stopping = early_stopping
    self.validation_fraction = validation_fraction
    self.n_iter_no_change = n_iter_no_change
    self.warm_start = warm_start
    self.average = average
    self.weights = None 
    self.bias = None

  def fit(self, X, y):
    pass 

  def predict(self, X):
    pass 

  def _get_loss(self, ):
    if self.loss == 'squared_loss':
      return mean_squared_error
    elif self.loss == 'huber':
      return huber
    elif self.loss == 'epsilon_insensitive':
      return epsilon_insensitive
    elif self.loss == 'squared_epsilon_insensitive':
      return squared_epsilon_insensitive
    else:
      raise ValueError('Loss function should be either "squared_loss", "huber", "epsilon_insensitive", or "squared_epsilon_insensitive"')