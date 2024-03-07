from core.BaseEstimator import BaseEstimator
from metrics.ClassificationMetrics import accuracy_score
import numpy as np

class LinearDiscriminantAnalysis(BaseEstimator):

  '''
  Linear Discriminant Analysis (LDA)

  sklearn: A classifer with linear decision boundary, generated by fitting class conditional densities to the data and using Bayes' rule.
  '''
  def __init__(self):
    super().__init__('LDA', 'Discriminant Analysis', accuracy_score)

  
  def fit(self, X, y):
    pass 

  def predict(self, X):
    pass 

  def evaluate(self, X, y, metric):
    pass

  

