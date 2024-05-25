from core.BaseEstimator import BaseEstimator
from utils.ClassificationMetrics import accuracy_score
from utils.Functions import sigmoid, softmax
import numpy as np


class LogisticRegression(BaseEstimator):
    """
    Logistic Regression

    A regular logistic regression model that inherits from the BaseEstimator class.
    """

    def __init__(
        self,
        penalty="l2",
        tol=1e-4,
        alpha=1.0,
        max_iter=1000,
        l1_ratio=0.5,
        learning_rate=1e-4,
    ):
        """
        Initialize the logistic regression model with the penalty, tolerance, regularization strength, and other parameters
        params:
        - penalty: str: the penalty term to use for regularization
        - tol: float: the tolerance for the stopping criterion
        - alpha: float: the regularization strength
        - max_iter: int: the maximum number of iterations
        - l1_ratio: float: the ratio of L1 regularization to L2 regularization
        - learning_rate: float: the learning rate for the gradient descent algorithm
        """
        super().__init__("Logistic Regressor", "linear_model", accuracy_score)
        self.penalty = penalty
        assert self.penalty in [
            "l1",
            "l2",
            "elasticnet",
            "none",
        ], "penalty must be one of 'l1', 'l2', 'elasticnet', 'none'"
        self.tol = tol
        self.alpha = alpha
        assert self.alpha >= 0, "alpha must be greater than or equal to 0"
        self.max_iter = max_iter
        self.l1_ratio = l1_ratio  # only used when penalty is 'elasticnet'
        assert (
            self.l1_ratio is None or 0 <= self.l1_ratio <= 1
        ), "l1_ratio must be between 0 and 1"
        self.learning_rate = learning_rate

    def fit(self, X, y):

        # Check if the input features and target values are numpy arrays
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        # Get the number of samples, features and classes
        _, n_features = X.shape
        n_classes = len(np.unique(y))

        # Initialize the weights
        self.w = (
            np.random.randn(n_features, n_classes) * 0.01
        )  # Initialize the weights with small random values
        self.b = np.zeros(n_classes)

        # Fit the model
        for i in range(self.max_iter):
            dot_product = (
                X @ self.w + self.b
            )  # Calculate the dot product of X and the weights
            proba = softmax(
                dot_product
            )  # Calculate the probabilities using the softmax function
            error = (
                proba - np.eye(n_classes)[y.astype(int)]
            )  # Calculate the error (derivative of the loss function)
            gradient = X.T @ error
            gradient = self.regularization_gradient(gradient)
            self.w -= gradient * self.learning_rate
            self.b -= np.sum(error, axis=0) * self.learning_rate

            if (
                np.linalg.norm(gradient) < self.tol
            ):  # Check if the gradient is less than the tolerance
                break

        print("Logistic Regression model fitted successfully")

    def regularization_gradient(self, gradient):
        if self.penalty == "l2":
            return gradient + self.alpha * self.w
        elif self.penalty == "l1":
            return gradient + self.alpha * np.sign(self.w)
        elif self.penalty == "elasticnet":
            return gradient + self.alpha * (
                self.l1_ratio * np.sign(self.w) + (1 - self.l1_ratio) * self.w
            )
        return gradient

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        dot_product = X @ self.w + self.b
        return softmax(dot_product)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def clone(self):
        return LogisticRegression(
            self.penalty,
            self.tol,
            self.C,
            self.fit_intercept,
            self.class_weight,
            self.max_iter,
            self.l1_ratio,
        )
