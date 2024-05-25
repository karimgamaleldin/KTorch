from autograd.engine import Tensor
from nn.Module import Module


class Flatten(Module):
    def __init__(self, start_dim=None, end_dim=-1):
        """
        Initialize the flatten layer
        """
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim, self.end_dim)

    def parameters(self):
        return []
