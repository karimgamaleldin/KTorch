import nn
from autograd.engine import Tensor
from core import KTorch
from nn.init import simpleUniformInitialization

class Conv1d(nn.Module):
  '''
  A class that represents a 1D convolutional layer in a neural network.
  '''

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, bias: bool=True, padding_mode: str='zeros'):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.padding_mode = padding_mode
    self.bias = bias
    self.shape = (out_channels, in_channels, kernel_size)
    self.weight = KTorch.tensor(simpleUniformInitialization(self.shape))



  def forward(self, x: Tensor) -> Tensor:
    '''
    Perform the forward pass of the 1D convolutional layer
    '''
    pass

  def parameters(self):
