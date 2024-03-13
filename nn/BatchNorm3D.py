from autograd.engine import Tensor
from core import KTorch
import nn

class BatchNorm3D(nn.Module):
  '''
  A class that represents the batch normalization layer, for 5D data 
  '''
  def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True):
    '''
    Initialize the batch normalization layer with the number of features
    params:
      num_features: int: the number of features
      eps: float: a small value to avoid division by zero
      momentum: float: the momentum for running statistics
      affine: bool: whether to include learnable affine parameters
      track_running_stats: bool: whether to track running statistics
    '''
    super().__init__()
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    self.affine = affine
    self.track_running_stats = track_running_stats
    self.running_mean = KTorch.zeros((1, num_features, 1, 1, 1))
    self.running_var = KTorch.ones((1, num_features, 1, 1, 1))
    self.gamma = KTorch.ones((1, num_features, 1, 1, 1)) if affine else None
    self.beta = KTorch.zeros((1, num_features, 1, 1, 1)) if affine else None

  def forward(self, x: Tensor) -> Tensor:
    '''
    Perform the forward pass of the batch normalization layer
    '''
    assert x.ndim == 5, "Input tensor must be 5D"
    # Check if training mode is on and if true track runing statistics else use running statistics
    if self.training:
      mean = KTorch.mean(x, axis=(0, 2, 3, 4), keepdims=True)
      var = KTorch.var(x, axis=(0, 2, 3, 4), keepdims=True)
      self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
      self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
    else:
      mean = self.running_mean
      var = self.running_var

    # Normalize the input
    x = (x - mean) / KTorch.sqrt(var + self.eps)

    # Apply the affine transformation
    if self.affine:
      x = x * self.gamma + self.beta

    return x 

  def parameters(self):
    '''
    Return the parameters of the batch normalization layer
    '''
    return [self.gamma, self.beta] if self.affine else []
