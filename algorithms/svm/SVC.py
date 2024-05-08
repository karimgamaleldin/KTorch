from core.BaseEstimator import BaseEstimator
from utils import accuracy_score, linear_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel
import numpy as np
from cvxopt import solvers, matrix

class SVC(BaseEstimator):
  '''
  Support Vector Machine Classifier

  A simple implementation of Support Vector Machine Classifier, used for binary classification tasks.
  '''

  def __init__(self, C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0):
    '''
    Constructor for SVC class
    
    Parameters:
    C (float): Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
    kernel (str): Specifies the kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 'rbf', 'sigmoid'.
    degree (int): Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
    gamma (str): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If 'auto', then 1/n_features will be used instead.
    coef0 (float): Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.

    '''
    super().__init__('Support Vector Machine Classifier', 'svm', accuracy_score)
    self.C = C
    assert C > 0, 'Invalid regularization parameter'
    self.kernel = kernel
    assert kernel in ['linear', 'poly', 'rbf', 'sigmoid'], 'Invalid kernel'
    self.degree = degree
    self.gamma = gamma
    assert gamma in ['auto', 'scale'] or (isinstance(gamma, (int, float)) and gamma > 0), 'Invalid gamma'
    self.coef0 = coef0

  def fit(self, X, y):
    # Convert X and y to numpy arrays
    if not isinstance(X, np.ndarray):
      X = np.array(X)
    if not isinstance(y, np.ndarray):
      y = np.array(y)

    # Convert the labels to 1 and -1
    y[y == 0] = -1 

    # Get the number of classes & features
    self.classes = np.unique(y)
    self.features = X.shape[1]

    # Define the gamma for the kernel functions that require it
    if self.gamma == 'auto':
      self.gamma = 1 / self.features
    elif self.gamma == 'scale':
      self.gamma = 1 / (self.features * X.var())

    # Fit the model to the data

    # Compute the kernel matrix
    K = self._kernel(X, X) # (n_samples, n_samples) - contains the dot product of all samples in the z space

    # Solve the dual optimization problem
    alphas = self._solve_qp(K, y)

    # Get the support vectors
    support_vec_idx = alphas > 1e-6 # Support vectors are the ones with non-zero alphas, due to numerical precision 1e-6 is used
    self.alphas = alphas[support_vec_idx]
    self.support_vectors = X[support_vec_idx]
    self.labels = y[support_vec_idx]

    # Compute the bias term
    temp = [self.labels[i] - np.sum(self.alphas * self.labels * K[support_vec_idx, i]) for i in range(len(self.alphas))]
    self.bias = np.mean(temp)

    print(f'Number of support vectors: {len(self.alphas)}')
    print('Model trained successfully')


  def _kernel(self, X, Y):
    if self.kernel == 'linear':
      return linear_kernel(X, Y)
    elif self.kernel == 'poly':
      return polynomial_kernel(X, Y, self.degree, self.gamma, self.coef0)
    elif self.kernel == 'rbf':
      return rbf_kernel(X, Y, self.gamma)
    elif self.kernel == 'sigmoid':
      return sigmoid_kernel(X, Y, self.gamma, self.coef0)
    
  def _solve_qp(self, K, y):
    n_samples = K.shape[0]

    # Define the matrices for the quadratic programming problem, needed formulation for the solver
    P = matrix(np.outer(y, y) * K) # (n_samples, n_samples) - 
    q = matrix(-np.ones((n_samples, 1))) # (n_samples, 1) - needed for the quadratic programming problem to convert the dual optimization maximization problem to a minimization problem, will be multiplied by the alphas
    G = matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)])) # (2 * n_samples, n_samples) - inequality constraints, alphas >= 0 and alphas <= C, here we define wheather greater than the constraint or less than the constraint
    h = matrix(np.hstack([np.zeros(n_samples), np.ones(n_samples) * self.C])) # (2 * n_samples, n_samples) - inequality constraints, alphas >= 0 and alphas <= C, here we define the numerical values of the constraints
    A = matrix(y.reshape(1, -1), (1, n_samples), 'd') # (1, n_samples) - equality constraint, sum of alphas * y = 0 - this is the A
    b = matrix(np.zeros(1)) # (1, 1) - equality constraint, sum of alphas * y = 0 - this is the 0
    solution = solvers.qp(P, q, G, h, A, b)
    alphas = np.ravel(solution['x'])
    return alphas

  def predict(self, X):
    # Convert X to numpy array
    if not isinstance(X, np.ndarray):
      X = np.array(X)

    # Predict the labels
    K = self._kernel(X, self.support_vectors)
    preds = np.dot(K, self.alphas * self.labels) + self.bias
    out = np.sign(preds)
    # out[out == -1] = 0

    return out

  def evaluate(self, X, y, metric=accuracy_score):
    preds = self.predict(X)
    return metric(y, preds)

  def score(self, X, y):
    return self.evaluate(X, y, self._base_metric)
  
  def clone(self):
    return SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0, tol=self.tol, max_iter=self.max_iter)