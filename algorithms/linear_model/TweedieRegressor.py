from core.BaseEstimator import BaseEstimator
from utils.RegressionMetrics import (
    mean_squared_error,
    huber,
    epsilon_insensitive,
    squared_epsilon_insensitive,
)
import numpy as np


class TweedieRegressor(BaseEstimator):
    """
    Tweedie Regressor class

    A Generalized Linear Model with a Tweedie distribution and a power link function.

    The Tweedie distribution is a family of distributions that includes the normal, Poisson, and gamma distributions as special cases.

    Algorithms included implicitly in this class are:
    - Normal Regression (power=0) - similar to Linear Regression
    - Poisson Regression (power=1)
    - Gamma Regression (power=2)
    - Inverse Gaussian Regression (power=3)
    - Compound Poisson Gamma Regression (power=(1, 2))
    """

    def __init__(
        self,
        *,
        power: float = 0,
        alpha: float = 0.0,
        fit_intercept: bool = True,
        link: str = "auto",
        max_iter: int = 1000,
        tol: float = 1e-4
    ):
        """
        Initialize the Tweedie Regressor

        Parameters:
          - power: float - the power parameter of the Tweedie distribution. Default is 0.
          - alpha: float - the regularization strength. Default is 0.0.
          - fit_intercept: bool - whether to fit an intercept term. Default is True.
          - link: str - the link function to use. Default is 'auto'.
          - max_iter: int - the maximum number of iterations. Default is 1000.
          - tol: float - the tolerance for the optimization. Default is 1e-4.
        """
        super().__init__("Tweedie Regressor", "linear_model", mean_squared_error)
        assert power in [
            0,
            1,
            2,
            3,
            (1, 2),
        ], "power must be one of 0, 1, 2, 3, or (1, 2)"
        assert alpha >= 0, "alpha must be non-negative"
        assert max_iter > 0, "max_iter must be positive"
        assert tol > 0, "tol must be positive"
        assert link in [
            "auto",
            "identity",
            "log",
        ], 'link must be one of "auto", "identity" or "log"'
        self.power = power
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.link = link
        self.max_iter = max_iter
        self.tol = tol

    def _link_function(self, x: np.ndarray) -> np.ndarray:
        pass

    def _inverse_link_function(self, x: np.ndarray) -> np.ndarray:
        pass

    def _tweedie_deviance(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    def _tweedie_gradient(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    def _tweedie_hessian(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Tweedie Regressor on the given data

        Parameters:
          - X: np.ndarray - the input data
          - y: np.ndarray - the target values

        Returns:
          - self: TweedieRegressor - the fitted Tweedie Regressor
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, metric=mean_squared_error
    ) -> float:
        preds = self.predict(X)
        return metric(y, preds)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.evaluate(X, y, self.base_metric)

    def clone(self):
        return TweedieRegressor(
            power=self.power,
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            link=self.link,
            max_iter=self.max_iter,
            tol=self.tol,
        )
