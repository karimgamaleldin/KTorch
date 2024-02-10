from autograd.engine import Tensor
from core import KTorch
import nn 

class Sigmoid(nn.Module):
  '''
  A class that represents the tanh activation function
  '''
  def forward(self, x: Tensor) -> Tensor:
    '''
    Perform the forward pass of the tanh activation function
    '''
    self.out = KTorch.sigmoid(x)
    return self.out

  
