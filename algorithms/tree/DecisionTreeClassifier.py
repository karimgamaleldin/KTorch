from core.BaseEstimator import BaseEstimator
from metrics.ClassificationMetrics import accuracy_score, gini_index, entropy
import numpy as np

class DecisionTreeClassifier(BaseEstimator):
  
    '''
    Decision Tree Classifier
    A classifer that uses a decision tree to go from observations about an item to conclusions about the item's target value.
    '''
    def __init__(self, criterion='gini', splitter='best', max_depth = None, min_samples_split: int =2, max_features=None, min_impurity_decrease=0, ccp_alpha=0.0):
      super().__init__('Decision Tree', 'Tree', accuracy_score)
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
  
    
    def fit(self, X, y):
      if self.max_features is None:
        self.max_features = X.shape[1]
      else:
        assert 0 < self.max_features <= X.shape[1], 'max_features should be in the range (0, n_features]' 
      # build the tree
      self.root = self._build(X, y) 
  
    def predict(self, X):
      pass 
  
    def evaluate(self, X, y, metric):
      pass

    def _build(self, X, y, depth=0):
      # Check if we reached the max depth, if all the labels are the same or if the number of samples is less than the minimum samples to split
      if depth == self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
        return Node(value = np.bincount(y).argmax()) # Return the most common label
      
      # Get the index of the random features we will choose from
      features = np.random.choice(X.shape[1], self.max_features, replace=False)

      best_feature, best_threshold, best_impurity = self._best_split(X, y, features)

      # If the best impurity is 0 or less than the minimum impurity decrease, return the most common label (no more splits)
      if best_impurity == 0 or best_impurity < self.min_impurity_decrease:
        return Node(value = np.bincount(y).argmax()) # Return the most common label
      
      left_mask = X[:, best_feature] <= best_threshold
      right_mask = ~left_mask
      left = self._build(X[left_mask], y[left_mask], depth+1)
      right = self._build(X[right_mask], y[right_mask], depth+1)
      node = Node(best_feature, left, right)
      return node
    
    def _best_split(self, X, y):
      if self.splitter == 'best':
        return self._best_splitter(X, y)
      else:
        return self._random_splitter(X, y)
      
    def _best_splitter(self, X, y, features):
      pass 

    def _random_splitter(self, X, y, features):
      pass 

    def clone(self):
      return DecisionTreeClassifier(self.criterion, self.splitter, self.max_depth, self.min_samples_split, self.max_features, self.min_impurity_decrease, self.ccp_alpha)
    

class Node:
    def __init__(self, feature, left=None, right=None, value =None):
        self.feature = feature
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
      return self.left is None and self.right is None
