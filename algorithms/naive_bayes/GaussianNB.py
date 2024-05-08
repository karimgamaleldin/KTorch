from core.BaseEstimator import BaseEstimator
import numpy as np
from utils.ClassificationMetrics import accuracy_score

class GaussianNB(BaseEstimator):
  '''
  Gaussian Naive Bayes
  
  An implementation of Gaussian Naive Bayes algorithm.
  '''
  def __init__(self, prior=None, var_smoothing=1e-9):
    super().__init__('Gaussian Naive Bayes', 'naive_bayes', accuracy_score)
    self.prior = prior
    self.var_smoothing = var_smoothing

  def fit(self, X, y):
    if not isinstance(X, np.ndarray):
      X = np.array(X)
    if not isinstance(y, np.ndarray):
      y = np.array(y)

    if self.prior is None:
      self.prior = np.bincount(y) / len(y)

    self.classes = np.unique(y) # unique classes
    self.n_classes, self.n_features = len(self.classes), X.shape[1] # number of classes and features
    self.log_prior = np.log(self.prior) # log of prior probabilities
    self.class_means = np.array([X[y == c].mean(axis=0) for c in self.classes]) # Calculate mean of each class
    self.class_var = np.array([X[y == c].var(axis=0) + self.var_smoothing for c in self.classes]) # Calculate variance of each class, add smoothing factor for numerical stability and better generalization
    return self

  def predict(self, X):
    n_samples = X.shape[0]
    posterior_log_likelihood = np.zeros((n_samples, self.n_classes)) # initialize posterior log likelihood (propotional to the posterior probability of each class given the data)
    for i in range(self.n_classes):
      # log of the coefficient of the Gaussian distribution
      log_prob_term1 = -0.5 * np.log(2 * np.pi * self.class_var[i]) 
      # log of the exponential term of the Gaussian distribution
      log_prob_term2 = -0.5 * ((X - self.class_means[i]) ** 2) / self.class_var[i] 
      # log of the prior probability of the class (needed to apply Bayes' theorem)
      log_prior = self.log_prior[i] 
      # calculate the posterior log likelihood
      posterior_log_likelihood[:, i] = (log_prob_term1 + log_prob_term2).sum(axis=1) + self.log_prior[i] # axis=1 sums the log probabilities of the assumed independent features
    # return the class with the highest posterior probability
    return self.classes[np.argmax(posterior_log_likelihood, axis=1)]
  
  def evaluate(self, X, y, metric=accuracy_score):
    preds = self.predict(X)
    return metric(y, preds)
  
  def score(self, X, y):
    return self.evaluate(X, y, self._base_metric)
  
  def clone(self):
    return GaussianNB(prior=self.prior, var_smoothing=self.var_smoothing)
  


