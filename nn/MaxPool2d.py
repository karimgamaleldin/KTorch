from autograd import Tensor
from core import KTorch
from nn import Module

class MaxPool2d(Module):
  '''
  A class that represents a 2D max pooling layer in a neural network.
  '''
  def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
    '''
    Initialize the 2D max pooling layer with the kernel size, stride, and padding
    params:
      kernel_size: int: the size of the kernel
      stride: int: the stride of the pooling
      padding: int: the padding to apply around the input
    '''
    super().__init__()
    self.kernel_size = kernel_size
    self.stride = stride if stride is not None else kernel_size
    self.padding = ((0, 0), (0, 0), (padding, padding), (padding, padding))

  def forward(self, x: Tensor) -> Tensor:
    '''
    Perform the forward pass of the 2D max pooling layer
    '''
    assert x.ndim == 4, "Input tensor must be 4D"
    # Pad the input tensor
    x = KTorch.pad(x, self.padding, mode='zeros')
    # Apply the max pooling operation
    return x.max_pool2d(self.kernel_size, self.stride)
  
  def parameters(self):
    '''
    Return the parameters of the 2D max pooling layer
    '''
    return []
