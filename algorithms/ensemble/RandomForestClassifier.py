from core.BaseEstimator import BaseEstimator
from algorithms.tree.DecisionTreeClassifier import DecisionTreeClassifier
from utils.ClassificationMetrics import gini_index, entropy, accuracy_score
from scipy.stats import mode
import numpy as np


class RandomForestClassifier(BaseEstimator):
    """
    Random Forest Classifier
    A classifier that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
    """

    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        max_features=None,
        min_impurity_decrease=0,
        bootstrap=True,
        oob_score=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        """
        Random Forest Classifier constructor

        Parameters:
          - n_estimators: Number of trees in the forest
          - criterion: The function to measure the quality of a split. Supported criteria are 'gini' for the Gini impurity and 'entropy' for the information gain
          - max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples
          - min_samples_split: The minimum number of samples required to split an internal node
          - max_features: The number of features to consider when looking for the best split
          - min_impurity_decrease: A node will be split if this split induces a decrease of the impurity greater than or equal to this value
          - bootstrap: Whether bootstrap samples are used when building trees
          - oob_score: Whether to use out-of-bag samples to estimate the generalization error
          - ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning
          - max_samples: The proportion of samples to draw from X to train each base estimator
        """
        super().__init__("Random Forest Classifier", "Ensemble", accuracy_score)
        self.n_estimators = n_estimators
        assert criterion in ["gini", "entropy"], "criterion should be gini or entropy"
        self.criterion = criterion
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

    def fit(self, X, y):
        if self.max_samples is None:
            self.max_samples = X.shape[0]
        else:
            assert (
                self.max_samples <= X.shape[0]
            ), "max_samples should be less than or equal to the number of samples"

        if self.oob_score:
            self.oob_predictions = {
                i: [] for i in range(X.shape[0])
            }  # initialize out-of-bag predictions object

        num_drawn_samples = min(int(X.shape[0] * self.max_samples), X.shape[0])
        for _ in range(self.n_estimators):
            # Create decision tree
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha,
            )
            # Create bootstrap sample
            if self.bootstrap:
                indices = np.random.choice(X.shape[0], num_drawn_samples, replace=True)
            else:
                indices = np.random.choice(X.shape[0], num_drawn_samples, replace=False)
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
            oob_indices = []
            for i, preds in self.oob_predictions.items():
                if len(preds) > 0:
                    oob_preds.append(mode(preds)[0])
                    oob_indices.append(i)

            oob_preds = np.array(oob_preds)
            oob_indices = np.array(oob_indices)

            print("Out-of-bag score:", accuracy_score(y[oob_indices], oob_preds))

        print("Random Forest Classifier fitted")

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return mode(predictions, axis=0)[0].ravel()

    def evaluate(self, X, y, metric):
        predictions = self.predict(X)
        return metric(y_true=y, y_pred=predictions)

    def score(self, X, y):
        return self.evaluate(X, y, self.base_metric)

    def clone(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
        )
