from core.algorithm_interfaces.DecisionTreeInterface import DecisionTreeInterface
from metrics.ClassificationMetrics import accuracy_score, gini_index, entropy
import numpy as np

class DecisionTreeClassifier(DecisionTreeInterface):
  
    '''
    Decision Tree Classifier
    A classifer that uses a decision tree to go from observations about an item to conclusions about the item's target class.
    '''
    def __init__(self, criterion='gini', splitter='best', max_depth = None, min_samples_split: int =2, max_features=None, min_impurity_decrease=0, ccp_alpha=0.0):
      self.root = None
      assert criterion in ['gini', 'entropy'], 'criterion should be gini or entropy or log_loss'
      self.criterion = gini_index if criterion == 'gini' else entropy
      assert splitter in ['best', 'random'], 'splitter should be best or random'
      self.splitter = splitter # done 
      self.max_depth = max_depth # done
      self.min_samples_split = min_samples_split # done
      assert  isinstance(max_features, int) or isinstance(max_features, float) or max_features in ['sqrt', 'log2'], 'max_features should be an integer or float or sqrt or log2'
      self.max_features = max_features # done
      self.min_impurity_decrease = min_impurity_decrease # done
      self.ccp_alpha = ccp_alpha
      super().__init__(self.criterion, self.splitter, self.max_depth, self.min_samples_split, self.max_features, self.min_impurity_decrease, self.ccp_alpha, 'classification')
  
    
    def fit(self, X, y):
      super().fit(X, y)
  
    def predict(self, X):
      super().predict(X) 
  
    def evaluate(self, X, y, metric):
      super().evaluate(X, y, metric)

    def clone(self):
      return DecisionTreeClassifier(self.criterion, self.splitter, self.max_depth, self.min_samples_split, self.max_features, self.min_impurity_decrease, self.ccp_alpha)
    