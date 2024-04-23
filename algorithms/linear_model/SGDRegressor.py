from core.BaseEstimator import BaseEstimator
from metrics.RegressionMetrics import mean_squared_error, huber, epsilon_insensitive, squared_epsilon_insensitive
import numpy as np

class SGDRegressor(BaseEstimator):
  '''
  Stochastic Gradient Descent Regressor
  
  A linear regression model that uses stochastic gradient descent to optimize the model parameters.

  It supports Ridge (L2), Lasso (L1) regularization and Elastic Net (L1 + L2) regularization.
  '''

  def __init__(self, loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, max_iter=1000, fit_intercept=True, learning_rate='invscaling', eta0=0.01, power_t=0.25, epsilon=0.1):
    '''
    Initialize the SGDRegressor model with the loss function, penalty, regularization strength, and other parameters
    params:
    - loss: str: the loss function to use
    - penalty: str: the penalty term to use for regularization
    - alpha: float: the regularization strength
    - l1_ratio: float: the ratio of L1 regularization to L2 regularization, for elastic net regularization
    - max_iter: int: the maximum number of iterations
    - fit_intercept: bool: whether to fit an intercept term
    - learning_rate: str: the learning rate schedule
    - eta0: float: the initial learning rate
    - power_t: float: the exponent for inverse scaling learning rate
    - epsilon: float: the epsilon value for epsilon-insensitive loss or huber loss
    '''
    super().__init__('SGD Regressor', 'linear_model', mean_squared_error)
    self.loss = loss
    assert self.loss in ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], "loss must be one of 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'"
    self.penalty = penalty
    assert self.penalty in ['l1', 'l2', 'elasticnet', 'none'], "penalty must be one of 'l1', 'l2', 'elasticnet', 'none'"
    self.alpha = alpha
    assert self.alpha >= 0, "alpha must be greater than or equal to 0"
    self.l1_ratio = l1_ratio
    assert 0 <= self.l1_ratio <= 1, "l1_ratio must be between 0 and 1"
    self.max_iter = max_iter
    assert self.max_iter > 0, "max_iter must be greater than 0"
    self.fit_intercept = fit_intercept
    self.learning_rate = learning_rate
    assert self.learning_rate in ['constant', 'optimal', 'invscaling', 'adaptive'], "learning_rate must be one of 'constant', 'optimal', 'invscaling', 'adaptive'"
    self.eta0 = eta0
    assert self.eta0 > 0, "eta0 must be greater than 0"
    self.power_t = power_t
    assert self.power_t >= 0, "power_t must be greater than or equal to 0"
    self.epsilon = epsilon
    assert self.epsilon >= 0, "epsilon must be greater than or equal to 0"
    self.coef_ = None
    self.intercept_ = None

  def fit(self, X, y):
    # Check if the input features and target values are numpy arrays
    if not isinstance(X, np.ndarray):
      X = X.to_numpy().astype(np.float64)
    if not isinstance(y, np.ndarray):
      y = y.to_numpy().astype(np.float64)

    # Initialize the weights
    n_samples, n_features = X.shape
    self.coef_ = np.random.randn(n_features, 1)
    self.intercept_ = 0.0

    # Fit the model
    prev_loss = np.inf
    for i in range(self.max_iter):
      for j in range(n_samples):
        self.t = i * n_samples + j
        X_j = X[j]
        y_j = y[j]
        y_pred = X_j @ self.coef_ + self.intercept_
        loss = self.loss_function(y_j, y_pred)
        gradient = self.gradient(y_j, y_pred)
        self.coef_ -= self.eta0 * (gradient * X_j.reshape(-1, 1) + self.regularization_gradient())
        self.intercept_ -= self.eta0 * gradient
        self.update_lr(t0=1.0, div=prev_loss<=loss)
        prev_loss = loss

    print('SGD Regressor model fitted successfully')


  def predict(self, X):
    return X @ self.coef_ + self.intercept_ 
  
  def evaluate(self, X, y, metric=None):
    if metric is None:
      metric = self.base_metric
    predictions = self.predict(X)
    return metric(y, predictions)

  def score(self, X, y):
    y_pred = self.predict(X)
    return mean_squared_error(y, y_pred) 

  def clone(self):
    return SGDRegressor(self.loss, self.penalty, self.alpha, self.l1_ratio, self.max_iter, self.tol, self.fit_intercept, self.learning_rate, self.eta0, self.power_t, self.early_stopping, self.epsilon, self.n_iter_no_change)

  def loss_function(self, y_true, y_pred):
    '''
    Compute the loss function for the given true and predicted values.
    '''
    loss = self.regularization_term()
    if self.loss == 'squared_loss':
      loss += mean_squared_error(y_true, y_pred) 
    elif self.loss == 'huber':
      loss += huber(y_true, y_pred, self.epsilon)
    elif self.loss == 'epsilon_insensitive':
      loss += epsilon_insensitive(y_true, y_pred, self.epsilon)
    elif self.loss == 'squared_epsilon_insensitive':
      loss += squared_epsilon_insensitive(y_true, y_pred, self.epsilon)
    return loss
  
  def gradient(self, y_true, y_pred):
    '''
    Compute the gradient of the loss function with respect to the prediction
    '''
    grad = self.regularization_gradient()
    if self.loss == 'squared_loss':
      grad += 2 * (y_pred - y_true)
    elif self.loss == 'huber':
      error = y_true - y_pred
      grad += self.epsilon * np.sign(error) if np.abs(error) <= self.epsilon else error
    elif self.loss == 'epsilon_insensitive':
      error = y_true - y_pred
      grad += self.epsilon * np.sign(error) if np.abs(error) <= self.epsilon else 0
    elif self.loss == 'squared_epsilon_insensitive':
      error = y_true - y_pred
      grad += 2 * self.epsilon * error if np.abs(error) <= self.epsilon else 0
    return grad
  
  def regularization_gradient(self):
    '''
    Compute the gradient of the regularization term with respect to the weights
    '''
    if self.penalty == 'l1':
      return self.alpha * np.sign(self.coef_)
    elif self.penalty == 'l2':
      return self.alpha * self.coef_
    return self.l1_ratio * np.sign(self.coef_) + (1 - self.l1_ratio) * self.coef_
  
  def regularization_term(self):
    '''
    Compute the regularization term for the weights
    '''
    if self.penalty == 'l1':
      return self.alpha * np.sum(np.abs(self.coef_))
    elif self.penalty == 'l2':
      return 0.5 * self.alpha * np.sum(self.coef_ ** 2)
    return self.l1_ratio * np.sum(np.abs(self.coef_)) + 0.5 * (1 - self.l1_ratio) * np.sum(self.coef_ ** 2)

  def update_lr(self, t0=1.0, div=False):
    '''
    Function to update the learning rate based on the learning rate schedule
    '''
    if self.learning_rate == 'constant':
      self.eta0 = self.eta0
    elif self.learning_rate == 'optimal':
      self.eta0 = 1.0 / (self.alpha * (t0 + self.t))
    elif self.learning_rate == 'invscaling':
      self.eta0 = self.eta0 / pow(t0 + self.t, self.power_t)
    elif self.learning_rate == 'adaptive':
      self.eta0 = self.eta0 if not div else self.eta0 / 5.0