from core.algorithm_interfaces.DecisionTreeInterface import DecisionTreeInterface
from utils.RegressionMetrics import (
    mean_squared_error,
    friedman_mse,
    mean_absolute_error,
    mean_poisson_deviance,
    r_squared,
)
import numpy as np


class DecisionTreeRegressor(DecisionTreeInterface):
    """
    Decision Tree Regressor
    A regressor that uses a decision tree to go from observations about an item to conclusions about the item's target value.
    """

    def __init__(
        self,
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split: int = 2,
        max_features=None,
        min_impurity_decrease=0,
        ccp_alpha=0.0,
    ):
        assert criterion in [
            "squared_error",
            "friedman_mse",
            "absolute_error",
            "poisson",
        ], "criterion should be squared_error, friedman_mse, absolute_error or poisson"
        if criterion == "squared_error":
            self.criterion = mean_squared_error
        elif criterion == "friedman_mse":
            self.criterion = friedman_mse
        elif criterion == "absolute_error":
            self.criterion = mean_absolute_error
        else:
            self.criterion = mean_poisson_deviance
        super().__init__(
            self.criterion,
            splitter,
            max_depth,
            min_samples_split,
            max_features,
            min_impurity_decrease,
            ccp_alpha,
            "regression",
            r_squared,
        )

    def fit(self, X, y):
        super().fit(X, y)

    def predict(self, X):
        return super().predict(X)

    def evaluate(self, X, y, metric):
        return super().evaluate(X, y, metric)

    def clone(self):
        return DecisionTreeRegressor(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
        )
