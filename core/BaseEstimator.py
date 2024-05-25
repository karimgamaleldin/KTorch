from utils.RegressionMetrics import mean_squared_error


class BaseEstimator:
    """
    Base class for all algorithms for example: linear regression, logistic regression, decision tree, random forest, etc.
    """

    def __init__(self, algorithm_name, algorithm_type, base_metric):
        self._algorithm_name = algorithm_name
        self._algorithm_type = algorithm_type
        self._base_metric = base_metric

    @property
    def algorithm_name(self):
        return self._algorithm_name

    @property
    def algorithm_type(self):
        return self._algorithm_type

    @property
    def base_metric(self):
        return self._base_metric

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters:
          - X: Input features (numpy array or pandas DataFrame)
          - y: Target values (numpy array or pandas Series)
        """
        raise NotImplementedError("Fit method must be implemented in the subclass")

    def predict(self, X):
        """
        Predict the target values for the input features.

        Parameters:
          - X: Input features (numpy array or pandas DataFrame)

        Returns:
          - y: Predicted target values (numpy array)
        """
        raise NotImplementedError("Predict method must be implemented in the subclass")

    def evaluate(self, X, y, metric=None):
        """
        Evaluate the model on the given test data.

        Parameters:
          - X: Input features (numpy array or pandas DataFrame)
          - y: Target values (numpy array or pandas Series)
          - metric: Evaluation metric (function)

        Returns:
          - score: Evaluation score (float)
        """
        if metric is None:
            metric = self.base_metric
        predictions = self.predict(X)
        return metric(y_true=y, y_pred=predictions)

    def clone(self):
        """
        Create a copy of the estimator.

        Returns:
          - estimator: A new instance of the estimator
        """
        raise NotImplementedError("Clone method must be implemented in the subclass")
