from core import KTorch
from nn.Module import Module
from autograd import Tensor
import numpy as np


class NLLLoss(Module):
    """
    NLLLoss

    A class that represents the negative log likelihood loss
    """

    def __init__(self, reduction: str = "mean"):
        """
        Initialize the nll loss

        reduction: str - The reduction method
        """
        super().__init__()
        assert reduction in ["mean", "sum"], "reduction must be either 'mean' or 'sum'"
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Forward pass

        y_pred: Tensor - The predicted values, the input should be the log probabilities
        y_true: Tensor - The true values, the input should be the class labels
        """

        batch_size = y_pred.shape[0]
        log_prob = y_pred[range(batch_size), y_true.view(-1).data.astype(int)]
        loss = -log_prob

        # Compute the reduction
        if self.reduction == "mean":
            return KTorch.mean(loss)
        elif self.reduction == "sum":
            return KTorch.sum(loss)

    def parameters(self):
        """
        Return the parameters
        """
        return []
