from autograd.engine import Tensor
from core import KTorch
from nn.Module import Module


class BatchNorm1D(Module):
    """
    A class that represents a batch normalization layer in a neural network.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        """
        Initialize the batch normalization layer with the number of features
        params:
          num_features: int: the number of features
          eps: float: a small value to avoid division by zero
          momentum: float: the momentum for running statistics
          affine: bool: whether to include learnable affine parameters
          track_running_stats: bool: whether to track running statistics
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.running_mean = KTorch.zeros((1, num_features))
        self.running_var = KTorch.ones((1, num_features))
        self.gamma = KTorch.ones((1, num_features)) if affine else None
        self.beta = KTorch.zeros((1, num_features)) if affine else None

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform the forward pass of the batch normalization layer
        """
        # Check if training mode is on and if true track runing statistics else use running statistics
        assert x.ndim == 2, "Input tensor must be 2D"
        ax = 0
        if self.training:
            mean = KTorch.mean(x, axis=ax, keepdims=True)
            var = KTorch.var(
                x, axis=ax, keepdims=True, unbiased=False
            )  # Biased variance like PyTorch
            if (
                self.track_running_stats
            ):  # Check if running statistics should be tracked
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * mean
                temp_var = KTorch.var(
                    x, axis=ax, keepdims=True, unbiased=True
                )  # Unbiased variance like PyTorch
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * temp_var
        else:
            if self.track_running_stats:
                mean = self.running_mean
                var = self.running_var
            else:
                assert (
                    x.shape[0] > 1
                ), "Batch size must be greater than 1 for inference mode without tracking running statistics"
                mean = KTorch.mean(x, axis=ax, keepdims=True)
                var = KTorch.var(
                    x, axis=ax, keepdims=True, unbiased=False
                )  # Biased variance like PyTorch

        # Normalize the input
        x = (x - mean) / KTorch.sqrt(var + self.eps)

        # Apply the affine transformation
        if self.affine:
            x = x * self.gamma + self.beta

        return x

    def parameters(self):
        """
        Return the parameters of the batch normalization layer
        """
        return [self.gamma, self.beta] if self.affine else []
