from autograd.engine import Tensor
from core import KTorch
from nn.Module import Module


class ReLU(Module):
    """
    A class that represents the ReLU activation function
    """

    def __init__(self):
        """
        Initialize the relu activation function
        """
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform the forward pass of the tanh activation function
        """
        self.out = KTorch.relu(x)
        return self.out
