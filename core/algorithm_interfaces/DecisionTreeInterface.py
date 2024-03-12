from core.BaseEstimator import BaseEstimator
import numpy as np

class DecisionTreeInterface(BaseEstimator):
  '''
  An interface for DecisionTreeRegressor and DecisionTreeClassifier, for reusability and consistency
  '''

  def __init__(self, criterion, splitter, max_depth, min_samples_split, max_features, min_impurity_decrease, ccp_alpha, type):
    super().__init__('Decision Tree', 'Tree', None)
    self.root = None
    self.criterion = criterion
    self.splitter = splitter
    self.max_depth = max_depth  
    self.min_samples_split = min_samples_split
    self.max_features = max_features
    self.min_impurity_decrease = min_impurity_decrease
    self.ccp_alpha = ccp_alpha
    self.type = type

  def fit(self, X, y):
    if self.max_features is None:
      self.max_features = X.shape[1]
    else:
      assert 0 < self.max_features <= X.shape[1], 'max_features should be in the range (0, n_features]'
    # build the tree
    self.root = self._build(X, y)

  def predict(self, X):
    return np.array([self._get_value(x, self.root) for x in X])

  def evaluate(self, X, y, metric):
    predictions = self.predict(X)
    return metric(y_true=y, y_pred=predictions)

  def _build(self, X, y, depth=0):
    # Check if we reached the max depth, if all the labels are the same or if the number of samples is less than the minimum samples to split
    if depth == self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
      return _Node(value = np.mean(y)) if self.type == 'regression' else _Node(value = np.bincount(y).argmax()) # Return the mean of the labels or the most common label, for middle condition mean and most common label are the same as any label as there is only one label
    
    # Get the index of the random features we will choose from
    features = np.random.choice(X.shape[1], self.max_features, replace=False)

    feature, threshold, impurity = self._split(X, y, features) # Get the best split depending on those features

    # If the best impurity is 0 or less than the minimum impurity decrease, return the mean of the labels or the most common label (no more splits)
    if impurity == 0 or impurity < self.min_impurity_decrease:
      return _Node(value = np.mean(y)) if self.type == 'regression' else _Node(value = np.bincount(y).argmax())
    
    # Split the data into left and right
    data_left_mask = X[:, feature] <= threshold
    data_right_mask = ~data_left_mask

    # Recursively build the left and right nodes
    left = self._build(X[data_left_mask], y[data_left_mask], depth+1)
    right = self._build(X[data_right_mask], y[data_right_mask], depth+1)
    # Create a node with the best feature and threshold and the left and right nodes
    node = _Node(feature=feature, threshold=threshold, left=left, right=right)
    return node
  
  def _get_value(self, x, node):
    '''
    Get the value of the node (the mean of the labels or the most common label) depending on the feature and threshold
    '''
    if node.is_leaf():
      return node.value
    if x[node.feature] <= node.threshold:
      return self._get_value(x, node.left)
    return self._get_value(x, node.right)
    


  def _split(self, X, y, features):
    '''
    Choose which type of split we will use (best or random)
    '''
    if self.splitter == 'best':
      return self._best_splitter(X, y, features)
    
    return self._random_splitter(X, y, features)

  def _best_splitter(self, X, y, features):
    best, best_idx, best_threshold = -1, None, None

    for feature in features:
      thresholds = np.unique(X[:, feature])

      # For each unique value of the feature, we will try to split the data and get the best impurity
      for threshold in thresholds:
        # For each threshold we will split the data into left and right
        data_left_mask = X[:, feature] <= threshold
        data_right_mask = ~data_left_mask
        # Get the impurity of the left and right data, the if statement is for classification and the else is for regression as for regression we need to get the impurity of the left and right data with the mean of the labels
        left_impurity = self.criterion(y[data_left_mask]) if self.type == 'classification' else self.criterion(y[data_left_mask], np.mean(y[data_left_mask]))
        right_impurity = self.criterion(y[data_right_mask]) if self.type == 'classification' else self.criterion(y[data_right_mask], np.mean(y[data_right_mask]))

        # Get the impurity weighted average of the split, the function with the min impurity will be the best split (max information gain)
        n = len(y)
        left_impurity *= len(y[data_left_mask]) / n
        right_impurity *= len(y[data_right_mask]) / n
        impurity = left_impurity + right_impurity

        # If the impurity is the best so far, update the best impurity, the best feature and the best threshold
        if best == -1 or impurity < best:
          best = impurity
          best_idx = feature
          best_threshold = threshold

    return best_idx, best_threshold, best

  def _random_splitter(self, X, y, features):
    # Choose a random feature and threshold and get the impurity of the split
    feature = np.random.choice(features)
    threshold = np.random.choice(np.unique(X[:, feature]))
    data_left_mask = X[:, feature] <= threshold
    data_right_mask = ~data_left_mask
    left_impurity = self.criterion(y[data_left_mask]) if self.type == 'classification' else self.criterion(y[data_left_mask], np.mean(y[data_left_mask]))
    right_impurity = self.criterion(y[data_right_mask]) if self.type == 'classification' else self.criterion(y[data_right_mask], np.mean(y[data_right_mask]))
    
    # Get the impurity weighted average of the split, the function with the min impurity will be the best split (max information gain)
    n = len(y)
    left_impurity *= len(y[data_left_mask]) / n
    right_impurity *= len(y[data_right_mask]) / n
    impurity = left_impurity + right_impurity
    return feature, threshold, impurity

  def clone(self):
    raise NotImplementedError('Clone method must be implemented in the subclass')


class _Node:
  def __init__(self, feature=None, threshold=None, left=None, right=None, value =None):
      self.feature = feature
      self.left = left
      self.right = right
      self.value = value

  def is_leaf(self):
    '''
    Check if the node is a leaf
    '''
    # A node is a leaf if it has no children and has a value
    return self.left is None and self.right is None
