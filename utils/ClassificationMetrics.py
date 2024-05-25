import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Compute the accuracy score.

    Parameters:
      - y_true: True target values (numpy array or pandas Series)
      - y_pred: Predicted target values (numpy array or pandas Series)

    Returns:
      - score: Accuracy score (float)
    """
    return np.mean(y_true == y_pred)


def binary_cross_entropy(y_true, y_pred):
    """
    Compute the binary cross-entropy loss.

    Parameters:
      - y_true: True target values
      - y_pred: Predicted target values

    Returns:
      - loss: Binary cross-entropy loss (float)
    """
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def categorical_cross_entropy(y_true, y_pred):
    """
    Compute the categorical cross-entropy loss.

    Parameters:
      - y_true: True target values
      - y_pred: Predicted target values

    Returns:
      - loss: Categorical cross-entropy loss (float)
    """
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)


def gini_index(y, sample_weight=None):
    """
    Compute the Gini index.
    Params:
      - y: target values
      - sample_weight: sample weights for weighted Gini index
    Returns:
      - gini index
    """
    return (
        1 - np.sum((np.bincount(y) / len(y)) ** 2)
        if sample_weight is None
        else 1
        - np.sum((np.bincount(y, weights=sample_weight) / np.sum(sample_weight)) ** 2)
    )


def entropy(y, sample_weight=None):
    """
    Compute the entropy.
    Params:
        - y: target values
        - sample_weight: sample weights for weighted Entropy
    Returns:
        - entropy
    """
    if sample_weight is None:
        p = np.bincount(y) / len(y)
    else:
        p = np.bincount(y, weights=sample_weight) / np.sum(sample_weight)

    # Avoiding log(0) by filtering out zero probabilities
    p = p[p > 0]

    return -np.sum(p * np.log2(p))
