from core.BaseEstimator import BaseEstimator
from core import KTorch
import nn 
import numpy as np


class LayerNorm(BaseEstimator):
  '''
  Layer Normalization

  Performs layer normalization on a mini-batch of input data. 
  
  Layer normalization is an alternative to batch normalization and can be used as a drop-in replacement for batch normalization. 
  The layer normalization is applied to the input data, and the output is computed using the following formula:

  y = gamma * (x - mean) / sqrt(var + eps) + beta

  where x is the input, mean and var are the mean and variance of x, gamma and beta are learnable parameters, and eps is a small value added to avoid division by zero.
  '''

  def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True):
    '''
    Initialize the layer normalization layer with the input dimension

    params:
      normalized_shape: int: the number of features
      eps: float: a small value to avoid division by zero
      elementwise_affine: bool: whether to include learnable affine parameters
      bias: bool: whether to include bias parameters
    '''
    super().__init__()
    if isinstance(normalized_shape, int):
      normalized_shape = [normalized_shape]
    self.normalized_shape = normalized_shape
    self.eps = eps
    self.elementwise_affine = elementwise_affine
    self.bias = bias
    self.gamma = KTorch.ones(normalized_shape) if elementwise_affine else None
    self.beta = KTorch.zeros(normalized_shape) if bias else None

  def forward(self, x):
    '''
    Perform the forward pass of the layer normalization layer
    '''
    assert x.shape[-len(self.normalized_shape):] == self.normalized_shape, "Input tensor must have the same shape as the normalized shape"
    mean = KTorch.mean(x, axis=list(range(-len(self.normalized_shape))), keepdims=True)
    var = KTorch.var(x, axis=list(range(-len(self.normalized_shape))), keepdims=True)
    x = (x - mean) / KTorch.sqrt(var + self.eps)
    if self.elementwise_affine:
      x = x * self.gamma + self.beta
    return x 

  def parameters(self):
    '''
    Return the parameters of the layer normalization layer
    '''
    return [self.gamma, self.beta] if self.elementwise_affine else []