from core.BaseEstimator import BaseEstimator
import numpy as np


class DecisionTreeInterface(BaseEstimator):
    """
    An interface for DecisionTreeRegressor and DecisionTreeClassifier, for reusability and consistency
    """

    def __init__(
        self,
        criterion,
        splitter,
        max_depth,
        min_samples_split,
        max_features,
        min_impurity_decrease,
        ccp_alpha,
        type,
        metric,
    ):
        super().__init__("Decision Tree", "Tree", None)
        self.root = None
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.metric = metric
        self.type = type

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None
    ):  # Added type annotations
        if self.max_features is None:
            self.max_features = X.shape[1]
        else:
            assert (
                0 < self.max_features <= X.shape[1]
            ), "max_features should be in the range (0, n_features]"  # Added error message
        # Build the tree
        self.root = self._build(X, y, sample_weight=sample_weight)

    def predict(self, X: np.ndarray) -> np.ndarray:  # Added type annotations
        return np.array([self._get_value(x, self.root) for x in X])

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, metric
    ) -> float:  # Added type annotations
        predictions = self.predict(X)
        return metric(y_true=y, y_pred=predictions)

    def _build(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int = 0,
        sample_weight: np.ndarray = None,
    ) -> "_Node":  # Added type annotations
        # Check if we reached the max depth, if all the labels are the same or if the number of samples is less than the minimum samples to split
        if (
            depth == self.max_depth
            or len(np.unique(y)) == 1
            or len(y) < self.min_samples_split
        ):
            if len(y) == 0:
                return _Node(value=0)
            return (
                _Node(value=np.mean(y))
                if self.type == "regression"
                else _Node(value=np.bincount(y).argmax())
            )

        # Get the index of the random features we will choose from
        features = np.random.choice(X.shape[1], self.max_features, replace=False)
        feature, threshold, impurity = self._split(
            X, y, features, sample_weight=sample_weight
        )

        # If the best impurity is 0 or less than the minimum impurity decrease, return the mean of the labels or the most common label (no more splits)
        if impurity == 0 or impurity < self.min_impurity_decrease:
            if len(y) == 0:
                return _Node(value=0)
            return (
                _Node(value=np.mean(y))
                if self.type == "regression"
                else _Node(value=np.bincount(y).argmax())
            )

        # Split the data into left and right
        data_left_mask = X[:, feature] <= threshold
        data_right_mask = ~data_left_mask

        # Recursively build the left and right nodes
        left = self._build(X[data_left_mask], y[data_left_mask], depth + 1)
        right = self._build(X[data_right_mask], y[data_right_mask], depth + 1)
        # Create a node with the best feature and threshold and the left and right nodes
        node = _Node(feature=feature, threshold=threshold, left=left, right=right)
        return node

    def _get_value(self, x: np.ndarray, node: "_Node"):  # Added type annotations
        """
        Get the value of the node (the mean of the labels or the most common label) depending on the feature and threshold
        """
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._get_value(x, node.left)
        return self._get_value(x, node.right)

    def _split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        features: np.ndarray,
        sample_weight: np.ndarray = None,
    ):  # Added type annotations
        """
        Choose which type of split we will use (best or random)
        """
        if self.splitter == "best":
            return self._best_splitter(X, y, features, sample_weight=sample_weight)
        return self._random_splitter(X, y, features)

    def _best_splitter(
        self,
        X: np.ndarray,
        y: np.ndarray,
        features: np.ndarray,
        sample_weight: np.ndarray = None,
    ):  # Added type annotations
        best, best_idx, best_threshold = -1, None, None

        for feature in features:
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                # Split the data into left and right
                data_left_mask = X[:, feature] <= threshold
                data_right_mask = ~data_left_mask
                # Calculate impurity
                left_impurity = (
                    self.criterion(y[data_left_mask], sample_weight=sample_weight)
                    if self.type == "classification"
                    else self.criterion(
                        y[data_left_mask],
                        np.mean(y[data_left_mask]) if len(y[data_left_mask]) > 0 else 0,
                    )
                )
                right_impurity = (
                    self.criterion(y[data_right_mask], sample_weight=sample_weight)
                    if self.type == "classification"
                    else self.criterion(
                        y[data_right_mask],
                        (
                            np.mean(y[data_right_mask])
                            if len(y[data_right_mask]) > 0
                            else 0
                        ),
                    )
                )

                # Weighted average impurity
                n = len(y)
                left_impurity *= len(y[data_left_mask]) / n
                right_impurity *= len(y[data_right_mask]) / n
                impurity = left_impurity + right_impurity

                # Check for best impurity
                if best == -1 or impurity < best:
                    best = impurity
                    best_idx = feature
                    best_threshold = threshold

        return best_idx, best_threshold, best

    def _random_splitter(
        self, X: np.ndarray, y: np.ndarray, features: np.ndarray
    ):  # Added type annotations
        feature = np.random.choice(features)
        threshold = np.random.choice(np.unique(X[:, feature]))
        data_left_mask = X[:, feature] <= threshold
        data_right_mask = ~data_left_mask
        left_impurity = (
            self.criterion(y[data_left_mask])
            if self.type == "classification"
            else self.criterion(
                y[data_left_mask],
                np.mean(y[data_left_mask]) if len(y[data_left_mask]) > 0 else 0,
            )
        )
        right_impurity = (
            self.criterion(y[data_right_mask])
            if self.type == "classification"
            else self.criterion(
                y[data_right_mask],
                np.mean(y[data_right_mask]) if len(y[data_right_mask]) > 0 else 0,
            )
        )

        n = len(y)
        left_impurity *= len(y[data_left_mask]) / n
        right_impurity *= len(y[data_right_mask]) / n
        impurity = left_impurity + right_impurity
        return feature, threshold, impurity

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.evaluate(X, y, self.metric)

    def clone(self):
        raise NotImplementedError("Clone method must be implemented in the subclass")


class _Node:
    def __init__(
        self,
        feature: int = None,
        threshold: float = None,
        left: "_Node" = None,
        right: "_Node" = None,
        value: float = None,
    ):  # Added type annotations
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:  # Added type annotation
        """
        Check if the node is a leaf
        """
        return self.left is None and self.right is None
