from core.algorithm_interfaces.DecisionTreeInterface import DecisionTreeInterface
from utils.ClassificationMetrics import accuracy_score, gini_index, entropy
import numpy as np

class DecisionTreeClassifier(DecisionTreeInterface):
  
    '''
    Decision Tree Classifier
    A classifer that uses a decision tree to go from observations about an item to conclusions about the item's target class.
    '''
    def __init__(self, criterion='gini', splitter='best', max_depth=None, min_samples_split: int = 2, max_features=None, min_impurity_decrease=0, ccp_alpha=0.0):
      assert criterion in ['gini', 'entropy'], 'criterion should be gini or entropy'
      criterion_function = gini_index if criterion == 'gini' else entropy
      
      super().__init__(criterion_function, splitter, max_depth, min_samples_split, max_features, min_impurity_decrease, ccp_alpha, 'classification', accuracy_score)

    def fit(self, X, y, sample_weight=None):
      super().fit(X, y, sample_weight=sample_weight)
  
    def predict(self, X):
      return super().predict(X) 
  
    def evaluate(self, X, y, metric):
      return super().evaluate(X, y, metric)

    def clone(self):
      return DecisionTreeClassifier(self.criterion, self.splitter, self.max_depth, self.min_samples_split, self.max_features, self.min_impurity_decrease, self.ccp_alpha)
    