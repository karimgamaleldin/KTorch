from core.BaseEstimator import BaseEstimator
from algorithms.tree.DecisionTreeClassifier import DecisionTreeClassifier
from metrics.ClassificationMetrics import gini_index, entropy, accuracy_score
from scipy.stats import mode
import numpy as np

class RandomForestClassifier(BaseEstimator):
  '''
  Random Forest Classifier
  A classifier that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
  '''

  def __init__(self, n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, max_features=None, min_impurity_decrease=0, bootstrap: bool=True, oob_score: bool=False, ccp_alpha=0.0, max_samples=None):
    self.n_estimators = n_estimators
    assert criterion in ['gini', 'entropy'], 'criterion should be gini or entropy or log_loss'
    self.criterion = gini_index if criterion == 'gini' else entropy    
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.max_features = max_features
    self.min_impurity_decrease = min_impurity_decrease
    self.bootstrap = bootstrap
    self.oob_score = oob_score
    self.ccp_alpha = ccp_alpha
    self.max_samples = max_samples
    self.trees = []
    self.oob_predictions = []
    super().__init__('Random Forest Classifier', 'Ensemble', None)

  def fit(self, X, y):
    if self.max_samples is None:
      self.max_samples = X.shape[0]
    else:
      assert self.max_samples <= X.shape[0], 'max_samples should be less than or equal to the number of samples'

    if self.oob_score:
      self.oob_predictions = {i: [] for i in range(X.shape[0])} # initialize out-of-bag predictions object

    for _ in range(self.n_estimators):
      # Create decision tree
      tree = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features, min_impurity_decrease=self.min_impurity_decrease, ccp_alpha=self.ccp_alpha)
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
          oob_preds.append(mode(preds, axis=1)[0][0])
      
      # Remove samples with no out-of-bag predictions
      oob_indices = [i for i, preds in self.oob_predictions.items() if len(preds) > 0]
      oob_preds = np.array(oob_preds)
      print('Out-of-bag score:', accuracy_score(y[oob_indices], oob_preds))

    print('Random Forest Classifier fitted')
      

  def predict(self, X):
    predictions = np.array([tree.predict(X) for tree in self.trees])
    return mode(predictions, axis=0)[0][0]


  def evaluate(self, X, y, metric):
    predictions = self.predict(X)
    return metric(y_true=y, y_pred=predictions)

  def clone(self):
    return RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion, max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features, min_impurity_decrease=self.min_impurity_decrease, bootstrap=self.bootstrap, oob_score=self.oob_score, ccp_alpha=self.ccp_alpha, max_samples=self.max_samples)

