from core.BaseEstimator import BaseEstimator
import numpy as np
from algorithms.tree.DecisionTreeRegressor import DecisionTreeRegressor


class GradientBoostingRegressor(BaseEstimator):
    """
    Gradient Boosting for regression
    """

    def __init__(
        self,
        learning_rate=0.1,
        n_estimators=100,
        max_depth=None,
        min_samples_split: int = 2,
        max_features=None,
        min_impurity_decrease=0,
        ccp_alpha=0.0,
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.F0 = None  # initial prediction
        self.trees = []
        self.tree_weights = []
        self.train_errors = []

    def fit(self, X, y):
        self.F0 = np.mean(y)
        Fm = self.F0
        for _ in range(self.n_estimators):
            # compute the pseudo-residuals
            r = y - Fm
            # fit a regression tree to the residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha,
            )
            tree.fit(X, r)
            # update the prediction
            Fm += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        Fm = self.F0
        for tree in self.trees:
            Fm += self.learning_rate * tree.predict(X)
        return Fm

    def evaluate(self, X, y, metric):
        y_pred = self.predict(X)
        return metric(y, y_pred)

    def clone(self):
        return GradientBoostingRegressor(
            loss=self.loss,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            alpha=self.alpha,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
            tol=self.tol,
        )
