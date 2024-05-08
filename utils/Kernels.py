import numpy as np
import scipy as sp

def linear_kernel(X, Y):
  '''
  Compute the linear kernel between X and Y.
  K(x, y) = x^T y
  '''
  return X @ Y.T

def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
  '''
  Compute the polynomial kernel between X and Y.
  K(x, y) = (gamma x^T y + coef0)^degree
  - gamma: Hyperparameter of the polynomial kernel, specify how much influence a single training example has depending on its distance from the test example.
  - coef0: Independent term in kernel function, helps to shift the decision boundary to the desired direction making the model more flexible and fit polynomials up to the degree specified. (adds interaction terms to the model)
  '''
  if gamma is None:
    gamma = 1 / X.shape[1]
  return (gamma * X @ Y.T + coef0) ** degree

def rbf_kernel(X, Y, gamma=None):
  '''
  Compute the radial basis function (RBF) kernel between X and Y.
  K(x, y) = exp(-gamma ||x-y||^2)
  - gamma: Hyperparameter of the RBF kernel, specify how much influence a single training example has depending on its distance from the test example.
  '''
  if gamma is None:
    gamma = 1 / X.shape[1]
  # Squared Euclidean distance can be computed as ||x-y||^2 = x^T x - 2 x^T y + y^T y
  X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1) # Squared L2 norm of X
  Y_sq = np.sum(Y ** 2, axis=1).reshape(1, -1) # Squared L2 norm of Y
  dist = X_sq - 2 * (X @ Y.T) + Y_sq # Squared Euclidean distance
  return np.exp(-gamma * dist)

def sigmoid_kernel(X, Y, gamma=None, coef0=1):
  '''
  Compute the sigmoid kernel between X and Y.
  K(x, y) = tanh(gamma x^T y + coef0)
  - gamma: Hyperparameter of the sigmoid kernel, specify how much influence a single training example has depending on its distance from the test example.
  - coef0: Independent term in kernel function, helps to shift the decision boundary of the tanh horizontaly so change the location of the decision boundary.
  '''
  if gamma is None:
    gamma = 1 / X.shape[1]
  return np.tanh(gamma * X @ Y.T + coef0)