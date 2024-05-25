from optim.Optim import Optim
from autograd.engine import Tensor
from typing import List
import numpy as np


class Adadelta(Optim):
    """
    The Adadelta optimizer inspired by the PyTorch api
    """

    def __init__(self, params: List[Tensor], lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        """
        Initialize the optimizer with the parameters and the learning rate
        params:
          params: list: the parameters of the model
          lr: float: the learning rate of the optimizer
          rho: float: the smoothing constant, used for calculating the moving average
          eps: float: a term added to the denominator to improve numerical stability
          weight_decay: float: the weight decay (L2 penalty)
        """
        super().__init__(params)
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self.accumulator_value = [
            np.zeros_like(param.data) for param in self.params
        ]  # store the leaky average of the second moment of the gradient
        self.delta_accumulator = [
            np.zeros_like(param.data) for param in self.params
        ]  # store the leaky average of the second moment of the change in the parameters

    def step(self):
        """
        Update the parameters of the model
        """

        for i, param in enumerate(self.params):
            # Temporary variable to store the gradient
            grad = param.grad
            # Apply weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            # Update the accumulator value
            self.accumulator_value[i] = (
                self.rho * self.accumulator_value[i] + (1 - self.rho) * grad**2
            )
            # Compute the RMS
            rms_grad = np.sqrt(self.accumulator_value[i] + self.eps)
            rms_delta_x = np.sqrt(self.delta_accumulator[i] + self.eps)
            # calculate delta x
            delta_x = (rms_delta_x / rms_grad) * grad
            # update the delta accumulator
            self.delta_accumulator[i] = (
                self.rho * self.delta_accumulator[i] + (1 - self.rho) * delta_x**2
            )
            # update the parameter
            param.data -= self.lr * delta_x
