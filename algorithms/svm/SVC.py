from core import BaseEstimator
from metrics.ClassificationMetrics import accuracy_score
import numpy as np

class SVC(BaseEstimator):
  '''
  Support Vector Machine Classifier

  A simple implementation of Support Vector Machine Classifier
  '''

  def __init__(self, C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, tol=1e-3, max_iter=1000):
    '''
    Constructor for SVC class
    
    Parameters:
    C (float): Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
    kernel (str): Specifies the kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 'rbf', 'sigmoid'.
    degree (int): Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
    gamma (str): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If 'auto', then 1/n_features will be used instead.
    coef0 (float): Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
    tol (float): Tolerance for stopping criterion.
    max_iter (int): Hard limit on iterations within solver.
    '''
    super().__init__('Support Vector Machine Classifier', 'svm', accuracy_score)
    self.C = C
    assert C > 0, 'Invalid regularization parameter'
    self.kernel = kernel
    assert kernel in ['linear', 'poly', 'rbf', 'sigmoid'], 'Invalid kernel'
    self.degree = degree
    self.gamma = gamma
    assert gamma in ['auto', 'scale'], 'Invalid gamma'
    self.coef0 = coef0
    self.tol = tol
    self.max_iter = max_iter

  def fit(self, X, y):
    # Convert X and y to numpy arrays
    if not isinstance(X, np.ndarray):
      X = np.array(X)
    if not isinstance(y, np.ndarray):
      y = np.array(y)

    # Get the number of classes & features
    self.classes = np.unique(y)
    self.features = X.shape[1]

    # Fit the model to the data

  def predict(self, X):
    pass 

  def evaluate(self, X, y, metric=accuracy_score):
    preds = self.predict(X)
    return metric(y, preds)

  def score(self, X, y):
    return self.evaluate(X, y, self._base_metric)
  
  def clone(self):
    return SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0, tol=self.tol, max_iter=self.max_iter)