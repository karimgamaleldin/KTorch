import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid function.

    Parameters:
      - x: Input value

    Returns:
      - y: Output value
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    Compute the softmax function.

    Parameters:
      - x: Input value

    Returns:
      - y: Output value
    """
    exps = np.exp(
        x - np.max(x, axis=-1, keepdims=True)
    )  # subtract the maximum value to avoid numerical instability
    return exps / np.sum(exps, axis=-1, keepdims=True)
