from autograd.engine import Tensor
from core import KTorch
import nn
from nn.init import simpleUniformInitialization

class Conv2d(nn.Module):
  '''
  A class that represents a 2D convolutional layer in a neural network.
  ''' 

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True, padding_mode: str = 'zeros'):
    '''
    Initialize the 2D convolutional layer with the number of input channels, output channels, kernel size, stride, padding, and bias
    params:
      in_channels: int: the number of input channels
      out_channels: int: the number of output channels
      kernel_size: int: the size of the kernel
      stride: int: the stride of the convolution
      padding: int: the padding to apply around the input
      bias: bool: whether to include a bias term
      padding_mode: str: the padding mode to use
    '''
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    assert padding_mode in ['zeros', 'reflect', 'replicate', 'circular'], 'Invalid padding mode'
    self.padding_mode = padding_mode
    self.weight = simpleUniformInitialization((out_channels, in_channels, kernel_size, kernel_size))
    self.bias = simpleUniformInitialization((out_channels, 1, 1, 1)) if bias else None

  def forward(self, x: Tensor) -> Tensor:
    '''
    Perform the forward pass of the 2D convolutional layer
    '''
    assert x.ndim == 4, "Input tensor must be 4D"
    # Pad the input tensor
    x = KTorch.pad(x, self.padding, mode=self.padding_mode)

    # Apply the convolution operation
    out = x.conv2d(self.weight, self.stride, self.padding)
    # Add the bias term
    if self.bias is not None:
      out += self.bias
    return out
  
  
  def parameters(self):
    '''
    Return the parameters of the 2D convolutional layer
    '''
    return [self.weight, self.bias] if self.bias is not None else [self.weight]
