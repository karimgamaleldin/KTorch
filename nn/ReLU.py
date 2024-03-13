from autograd.engine import Tensor
from core import KTorch
import nn

class ReLU(nn.Module):
  '''
  A class that represents the tanh activation function
  '''
  def __init__(self):
    '''
    Initialize the relu activation function
    '''
    super().__init__()


  def forward(self, x: Tensor) -> Tensor:
    '''
    Perform the forward pass of the tanh activation function
    '''
    self.out = KTorch.relu(x)
    return self.out

  
