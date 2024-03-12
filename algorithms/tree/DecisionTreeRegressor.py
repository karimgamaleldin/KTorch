from core.algorithm_interfaces.DecisionTreeInterface import DecisionTreeInterface
from metrics.RegressionMetrics import mean_squared_error, friedman_mse, mean_absolute_error, mean_poisson_deviance
import numpy as np

class DecisionTreeRegressor(DecisionTreeInterface):
  '''
  Decision Tree Regressor
  A regressor that uses a decision tree to go from observations about an item to conclusions about the item's target value.
  '''

  def __init__(self, criterion='squared_error', splitter='best', max_depth = None, min_samples_split: int =2, max_features=None, min_impurity_decrease=0, ccp_alpha=0.0):
      self.root = None
      assert criterion in ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], 'criterion should be gini or entropy or log_loss'
      if criterion == 'squared_error':
        self.criterion = mean_squared_error
      elif criterion == 'friedman_mse':
        self.criterion = friedman_mse
      elif criterion == 'absolute_error':
        self.criterion = mean_absolute_error
      else:
        self.criterion = mean_poisson_deviance
      assert splitter in ['best', 'random'], 'splitter should be best or random'
      self.splitter = splitter # done 
      self.max_depth = max_depth # done
      self.min_samples_split = min_samples_split # done
      assert  isinstance(max_features, int) or isinstance(max_features, float) or max_features in ['sqrt', 'log2'], 'max_features should be an integer or float or sqrt or log2'
      self.max_features = max_features # done
      self.min_impurity_decrease = min_impurity_decrease # done
      self.ccp_alpha = ccp_alpha
      super().__init__(self.criterion, self.splitter, self.max_depth, self.min_samples_split, self.max_features, self.min_impurity_decrease, self.ccp_alpha, 'regression')

    
  def fit(self, X, y):
    super().fit(X, y)

  def predict(self, X):
    super().predict(X)

  def evaluate(self, X, y, metric):
    super().evaluate(X, y, metric)

  def clone(self):
    return DecisionTreeRegressor(criterion=self.criterion, splitter=self.splitter, max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features, min_impurity_decrease=self.min_impurity_decrease, ccp_alpha=self.ccp_alpha)
  