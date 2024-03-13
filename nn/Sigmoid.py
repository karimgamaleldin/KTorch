from autograd.engine import Tensor
from core import KTorch
import nn 

class Sigmoid(nn.Module):
  '''
  A class that represents the sigmoid activation function
  '''

  def __init__(self):
    '''
    Initialize the sigmoid activation function
    '''
    super().__init__()
    
  def forward(self, x: Tensor) -> Tensor:
    '''
    Perform the forward pass of the sigmoid activation function
    '''
    self.out = KTorch.sigmoid(x)
    return self.out

  
