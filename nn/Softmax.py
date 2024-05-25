from autograd import Tensor
from nn.Module import Module
from core import KTorch


class Softmax(Module):
    """
    Softmax

    A class that represents the softmax activation function
    """

    def __init__(self, dim=None):
        """
        Initialize the softmax activation function
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        x: Tensor - The input tensor
        """
        return KTorch.softmax(x, axis=-1)

    def parameters(self):
        """
        Return the parameters
        """
        return []
