from core.BaseEstimator import BaseEstimator
from klearn.tree.DecisionTreeRegressor import DecisionTreeRegressor
import numpy as np
from metrics.RegressionMetrics import mean_squared_error, friedman_mse, mean_absolute_error, mean_poisson_deviance

class RandomForestRegressor(BaseEstimator):
  '''
  Random Forest Regressor
  A regressor that fits a number of decision tree regressors on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
  '''

  def __init__(self, n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, max_features=None, min_impurity_decrease=0, bootstrap: bool=True, oob_score: bool=False, ccp_alpha=0.0, max_samples=None):
    self.n_estimators = n_estimators
    assert criterion in ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], 'criterion should be squared_error, friedman_mse, absolute_error or poisson'
    if criterion == 'squared_error':
      self.criterion = mean_squared_error
    elif criterion == 'friedman_mse':
      self.criterion = friedman_mse
    elif criterion == 'absolute_error':
      self.criterion = mean_absolute_error
    else:
      self.criterion = mean_poisson_deviance
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.max_features = max_features
    self.min_impurity_decrease = min_impurity_decrease
    self.bootstrap = bootstrap
    self.oob_score = oob_score
    self.ccp_alpha = ccp_alpha
    self.max_samples = max_samples
    self.trees = []
    self.oob_predictions = {}
    super().__init__('Random Forest Regressor', 'Ensemble', None)

  def fit(self, X, y):

    if self.max_samples is None:
      self.max_samples = X.shape[0]
    else:
      assert self.max_samples <= X.shape[0], 'max_samples should be less than or equal to the number of samples'

    # initialize out-of-bag predictions object
    if self.oob_score:
      self.oob_predictions = {i: [] for i in range(X.shape[0])} # initialize out-of-bag predictions object

    for _ in range(self.n_estimators):
      # Create decision tree 
      tree = DecisionTreeRegressor(criterion=self.criterion, max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features, min_impurity_decrease=self.min_impurity_decrease, ccp_alpha=self.ccp_alpha)
      # Create bootstrap sample
      if self.bootstrap:
        indices = np.random.choice(X.shape[0], self.max_samples, replace=True)
        predictors = X[indices]
        labels = y[indices]
      else:
        indices = np.random.choice(X.shape[0], self.max_samples, replace=False)
        predictors = X[indices]
        labels = y[indices]
      # Fit decision tree
      tree.fit(predictors, labels)
      # Save decision tree
      self.trees.append(tree)

      if self.oob_score:
        oob_indices = np.array(list(set(range(X.shape[0])) - set(indices)))
        oob_preds = tree.predict(X[oob_indices])
        for i, pred in zip(oob_indices, oob_preds):
          self.oob_predictions[i].append(pred)

    if self.oob_score:
      oob_preds = []
      for i, preds in self.oob_predictions.items():
        if len(preds) > 0:
          oob_preds.append(np.mean(preds))

      # Remove samples with no out-of-bag predictions
      oob_indices = [i for i, preds in self.oob_predictions.items() if len(preds) > 0]
      oob_predictions = np.array(oob_preds)

      print('Out-of-bag score:', self.criterion(y[oob_indices], self.oob_predictions))      


    print('Random Forest Regressor fitted')


  def predict(self, X):
    predictions = []
    for tree in self.trees:
      predictions.append(tree.predict(X))
    return np.mean(predictions, axis=0)

  def evaluate(self, X, y, metric):
    predictions = self.predict(X)
    return metric(y, predictions, metric)

  def clone(self):
    return RandomForestRegressor(n_estimators=self.n_estimators, criterion=self.criterion, max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features, min_impurity_decrease=self.min_impurity_decrease, bootstrap=self.bootstrap, oob_score=self.oob_score, ccp_alpha=self.ccp_alpha, max_samples=self.max_samples)
  
