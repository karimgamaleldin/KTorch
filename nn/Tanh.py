from autograd.engine import Tensor
from core import KTorch
from nn.Module import Module


class Tanh(Module):
    """
    A class that represents the tanh activation function
    """

    def __init__(self):
        """
        Initialize the tanh activation function
        """
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform the forward pass of the tanh activation function
        """
        self.out = KTorch.tanh(x)
        return self.out
