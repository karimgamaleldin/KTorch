from core.BaseEstimator import BaseEstimator
import numpy as np
from metrics.ClassificationMetrics import accuracy_score
from scipy.sparse import issparse

class MultinomialNB(BaseEstimator):
  '''
  Multinomial Naive Bayes
  
  An implementation of Multinomial Naive Bayes algorithm.
  '''
  def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
    '''
    Constructor for MultinomialNB class.
    
    Parameters:
    alpha (float): Smoothing parameter. Default is 1.0.
    fit_prior (bool): Whether to learn class prior probabilities. Default is True.
    class_prior (array-like): Prior probabilities of the classes. If specified, the priors are not adjusted according to the data. Default is None.
    '''
    super().__init__('Multinomial Naive Bayes', 'naive_bayes', accuracy_score)
    self.alpha = alpha
    self.fit_prior = fit_prior
    self.class_prior = class_prior

  def fit(self, X, y):
    '''
    Fit the model to the given data.
    X contain the count of each feature in each sample.
    '''
    # Convert X and y to numpy arrays
    if issparse(X):
      X = X.todense()
    if not isinstance(X, np.ndarray):
      X = np.array(X)
    if not isinstance(y, np.ndarray):
      y = np.array(y)

    # Get the number of classes & features
    self.classes = np.unique(y)
    self.features = X.shape[1]

    # Getting the prior probabilities of the classes
    if self.fit_prior:
      if self.class_prior is not None:
        self.class_prior = np.array(self.class_prior)
      else:
        self.class_prior = np.bincount(y) / len(y)
      
    else: 
      self.class_prior = np.ones(self.classes) / len(self.classes)

    # Getting the likelihood probabilities
    self.likelihood = np.zeros((len(self.classes), self.features))
    for i, c in enumerate(self.classes):
      X_c = X[y == c] # Get all the samples that are members of class c
      self.likelihood[i] = (np.sum(X_c, axis=0) + self.alpha) / (np.sum(X_c) + self.alpha * self.features) # Calculate the likelihood probabilities of each feature given class c

    print('MultiNB fit complete.')

  def predict(self, X):
    # Convert X to a numpy array
    if issparse(X):
      X = X.todense() 
    if not isinstance(X, np.ndarray):
      X = np.array(X)

    # Initialize the predictions array
    preds = np.zeros(X.shape[0])

    # Calculate the log probabilities for numerical stability
    log_probs = np.zeros((len(self.classes), X.shape[0]))

    for i, _ in enumerate(self.classes):
      log_probs[i] = np.log(self.class_prior[i]) + np.sum(np.log(self.likelihood[i]) * X.T, axis=0) # log_probs = log(prior) + sum(log(likelihood of each feature given class c)) * feature counts in X

    # Get the class with the maximum log probability
    preds = np.argmax(log_probs, axis=0)

    return preds

  def evaluate(self, X, y, metric=accuracy_score):
    preds = self.predict(X)
    return metric(y, preds)
  
  def score(self, X, y):
    return self.evaluate(X, y, self._base_metric)
  
  def clone(self):
    return MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior, class_prior=self.class_prior)