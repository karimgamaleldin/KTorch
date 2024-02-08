from nn.init import simpleUniformInitialization
from autograd.Tensor import Tensor
from core import KTorch
import nn

class Linear(nn.Module):
  '''
  A class that represents a linear layer in a neural network.
  '''

  def __init__(self, in_features, out_features, bias=True):
    '''
    Initialize the linear layer with the input and output features
    params:
      in_features: int: the number of input features
      out_features: int: the number of output features
      bias: bool: whether to include a bias term in the linear layer
    '''
    self.in_features = in_features
    self.out_features = out_features
    self.shape = (out_features, in_features)
    self.weight = KTorch.tensor(simpleUniformInitialization(self.shape)) # initialize the weights using the uniform distribution
    self.bias = KTorch.tensor(simpleUniformInitialization((1,out_features), out_shape=(out_features))) if bias else None # initialize the bias using the uniform distribution if bias is True

  def forward(self, x: Tensor) -> Tensor:
    '''
    Perform the forward pass of the linear layer
    out = x * W^T + b
    '''
    self.out = KTorch.matmul(x, self.weight.data.T)
    self.inp = x
    if self.bias is not None:
      self.out += self.bias 
    return self.out 

  def parameters(self):
    '''
    Return the parameters of the linear layer
    '''
    return [self.weight, self.bias] if self.bias is not None else [self.weight]