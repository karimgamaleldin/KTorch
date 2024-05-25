import numpy as np


def euclidean_distance(x1, x2, axis=1):
    """
    Compute the Euclidean distance between two points.

    Parameters:
      - x1: First point (numpy array)
      - x2: Second point (numpy array)

    Returns:
      - distance: Euclidean distance between the two points (float)
    """
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=axis))


def manhattan_distance(x1, x2, axis=1):
    """
    Compute the Manhattan distance between two points.

    Parameters:
      - x1: First point (numpy array)
      - x2: Second point (numpy array)

    Returns:
      - distance: Manhattan distance between the two points (float)
    """
    return np.sum(np.abs(x1 - x2), axis=axis)


def minkowski_distance(x1, x2, axis=1, p=2):
    """
    Compute the Minkowski distance between two points.

    Parameters:
      - x1: First point (numpy array)
      - x2: Second point (numpy array)
      - p: Power parameter for the Minkowski metric (int)

    Returns:
      - distance: Minkowski distance between the two points (float)
    """
    return np.sum(np.abs(x1 - x2) ** p, axis=axis) ** (1 / p)
