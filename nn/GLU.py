from core import KTorch
from autograd.engine import Tensor
import numpy as np
from nn.Module import Module 

class GLU(Module):
  '''
  A class to represent the Gated Linear Unit (GLU) activation function.

  The input tensor is split into two halves along the last dimension. The input tensor is assumed to have an even number of dimensions and projected to 2*dim dimensions,
  as the output tensor will be of size dim.
  '''

  def __init__(self, dim:int = -1):
    '''
    Constructor for the GLU class.
    '''
    super().__init__()
    self.dim = dim

  def forward(self, x:Tensor) -> Tensor:
    assert x.shape[self.dim] % 2 == 0, "The input tensor must have an even number of dimensions along the specified dimension."
    div = x.shape[self.dim] // 2
    a, b = x.split(div, dim=self.dim)
    return a * KTorch.sigmoid(b)

  def parameters(self):
    return []
