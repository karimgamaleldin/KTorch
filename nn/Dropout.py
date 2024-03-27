import nn
from core import KTorch
from autograd.engine import Tensor

class Dropout(nn.Module):
  '''
  A class that represents a dropout layer in a neural network.

  A regularization technique based on the paper Dropout: A Simple Way to Prevent Neural Networks from Overfitting by Srivastava et al. (2014)
  '''

  def __init__(self, p: float=0.5):
    '''
    Initialize the dropout layer with the dropout probability
    params:
      p: float: the dropout probability
    '''
    super().__init__()
    self.p = p

  def forward(self, x: Tensor) -> Tensor:
    '''
    Perform the forward pass of the dropout layer
    '''
    self.mask = KTorch.rand(x.shape) > self.p
    self.out = x * self.mask * (1.0 / (1.0 - self.p)) # scale the output by 1/(1-p) to maintain the expected value
    return self.out 

  def parameters(self):
    '''
    Return the parameters of the dropout layer
    '''
    return []