from autograd.Tensor import Tensor
import nn

class Flatten(nn.Module):
  def __init__(self, start_dim=1, end_dim=-1):
    self.start_dim = start_dim
    self.end_dim = end_dim

  def forward(self, x: Tensor) -> Tensor:
    return x.flatten(self.start_dim, self.end_dim)