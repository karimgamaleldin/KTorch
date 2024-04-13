from core import KTorch
from autograd.engine import Tensor
import numpy as np
from nn.Module import Module

class GELU(Module):
  '''
  A class that represents the GELU activation function.
  '''

  def __init__(self, approximate: str='none'):
    '''
    Initialize the GELU activation function.
    '''
    super().__init__()
    self.approximate = approximate
    assert self.approximate in ['none', 'tanh'], 'Invalid approximation method'

  def forward(self, x: Tensor) -> Tensor:
    if self.approximate == 'none':
      return x * x.phi()
    elif self.approximate == 'tanh':
      return 0.5 * x * (1 + KTorch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

  def parameters(self):
    '''
    Return the parameters of the GELU activation function.
    '''
    return []