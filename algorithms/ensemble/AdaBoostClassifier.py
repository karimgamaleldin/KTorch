from core.BaseEstimator import BaseEstimator
from metrics.ClassificationMetrics import accuracy_score
import numpy as np
from algorithms.tree.DecisionTreeClassifier import DecisionTreeClassifier

class AdaBoostClassifier(BaseEstimator):
  '''
  An AdaBoost classifier, using the SAMME algorithm (Stagewise Additive Modeling using a Multiclass Exponential loss function)

  It fits a classifier on the dataset and then fits copies of the classifer on weighted versions of the same dataset where we increase the weights of the incorrectly classified instances. 

  It is often used with decision trees, but can be used with any other classifier as well, however for simplicity, we will use a decision tree as the base estimator. (Other estimators will be added in the future)
  '''
  def __init__(self, n_estimators=50, learning_rate=1.0, epsilon=1e-10):
    '''
    Create a new AdaBoostClassifier
    params:
    estimator: the base estimator to fit on the dataset. If None, a DecisionTreeClassifier with max_depth=1 will be used
    n_estimators: the number of estimators to fit
    learning_rate: the learning rate, which controls the contribution of each estimator and the weights update
    '''
    super().__init__('AdaBoost', 'Ensemble', accuracy_score)
    self.n_estimators = n_estimators
    self.learning_rate = learning_rate
    self.epsilon = epsilon
    self.estimators = []
    self.estimator_weights = []
    self.estimator_errors = []

  def fit(self, X, y):
    '''
    Fit the AdaBoost classifier to the dataset using the SAMME algorithm
    '''
    # Number of samples
    n = X.shape[0]
    self.num_classes = len(np.unique(y))
    # Initialize weights to 1/n
    w = np.ones(n) / n
    for _ in range(self.n_estimators):
      # Fit a classifier with the current weights
      estimator = DecisionTreeClassifier(max_depth=1)
      estimator.fit(X, y, sample_weight=w)
      # Make predictions
      y_pred = estimator.predict(X)
      # Compute the weighted error
      error = np.sum(w * (y_pred != y)) / np.sum(w)
      if error <= 0:
        # If the error is 0, we can stop (Perfect fit, no need to continue)
        break
      # Compute the estimator weight
      estimator_weight = self.learning_rate * (np.log((1 - error + self.epsilon) / (error + self.epsilon)) + np.log(self.num_classes - 1))
      # Update the weights
      w *= np.exp(estimator_weight * (y_pred != y))
      w /= np.sum(w)
      # Save the estimator
      self.estimators.append(estimator)
      self.estimator_weights.append(estimator_weight)
      self.estimator_errors.append(error)

    print('AdaBoostClassifier fitted')

  def predict(self, X):
    # Class votes
    class_votes = np.zeros((X.shape[0], self.num_classes))

    for estimator, weight in zip(self.estimators, self.estimator_weights):
      predictions = estimator.predict(X)
      class_votes[range(X.shape[0]), predictions] += weight

    # Return the class with the highest vote
    weighted_predictions = np.argmax(class_votes, axis=1)

    return weighted_predictions 

  def evaluate(self, X, y, metric):
    predictions = self.predict(X)
    return metric(y_true=y, y_pred=predictions)

  def clone(self):
    return AdaBoostClassifier(estimator=self.base_estimator, n_estimators=self.n_estimators, learning_rate=self.learning_rate)
