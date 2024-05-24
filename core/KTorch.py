from autograd.engine import Tensor
import numpy as np 

def sigmoid(x: Tensor) -> Tensor:
  '''
  Perform the sigmoid activation function
  '''
  return x.sigmoid()

def tanh(x: Tensor) -> Tensor:
  '''
  Perform the tanh activation function
  '''
  return x.tanh()

def relu(x: Tensor) -> Tensor:
  '''
  Perform the ReLU activation function
  '''
  return x.ReLU()

def tensor(x) -> Tensor:
  '''
  Create a tensor
  '''
  return Tensor(x)

def zeros(shape) -> Tensor:
  '''
  Create a tensor of zeros
  '''
  return Tensor(np.zeros(shape))

def zeros_like(x: Tensor) -> Tensor:
  '''
  Create a tensor of zeros with the same shape as another tensor
  '''
  return Tensor(np.zeros_like(x.data))

def ones(shape) -> Tensor:
  '''
  Create a tensor of ones
  '''
  return Tensor(np.ones(shape))

def randn(shape) -> Tensor:
  '''
  Create a tensor of random numbers
  '''
  return Tensor(np.random.randn(*shape))

def matmul(x: Tensor, y: Tensor) -> Tensor:
  '''
  Perform the matrix multiplication of two tensors
  '''
  return x.__matmul__(y)

def sqrt(x: Tensor) -> Tensor:
  '''
  Perform the square root of a tensor
  '''
  return x.sqrt()

def mean(x: Tensor, axis=None, keepdims=False) -> Tensor:
  '''
  Compute the mean of a tensor
  '''
  return x.mean(axis=axis, keepdims=keepdims)

def var(x: Tensor, axis=None, keepdims=False, unbiased=False) -> Tensor:
  '''
  Compute the variance of a tensor
  '''
  return x.var(axis=axis, keepdims=keepdims, unbiased=unbiased)

def rand(shape) -> Tensor:
  '''
  Create a tensor of random numbers between 0 and 1
  '''
  return Tensor(np.random.rand(*shape))

def arange(start, stop=None, step=1, dtype=np.float32) -> Tensor:
  '''
  Create a tensor of evenly spaced values
  '''
  return Tensor(np.arange(start, stop, step))

def pow(x: Tensor, y: Tensor) -> Tensor:
  '''
  Compute the power of a tensor
  '''
  return x.__pow__(y)

def cos(x: Tensor) -> Tensor:
  '''
  Compute the cosine of a tensor
  '''
  return x.cos()

def sin(x: Tensor) -> Tensor:
  '''
  Compute the sine of a tensor
  '''
  return x.sin()

def phi(x: Tensor) -> Tensor:
  '''
  Compute the normal distribution CDF of a tensor
  '''
  return x.phi()

def softmax(x: Tensor, axis=None) -> Tensor:
  '''
  Compute the softmax of a tensor
  '''
  return x.softmax(axis=axis)


def pad(x: Tensor, pad_width, mode='zeros') -> Tensor:
  '''
  Pad a tensor
  '''
  return x.pad(pad_width, mode=mode)

def cat(tensors: list, axis=0) -> Tensor:
  '''
  Concatenate tensors
  '''
  return Tensor.cat(tensors, axis=axis)

  
def sum(x: Tensor, axis=None, keepdims=False) -> Tensor:
  '''
  Compute the sum of a tensor
  '''
  return x.sum(axis=axis, keepdims=keepdims)

def log(x: Tensor) -> Tensor:
  '''
  Compute the natural logarithm of a tensor
  '''
  return x.log()

def one_hot(x: Tensor, num_classes: int) -> Tensor:
  '''
  Create a one hot tensor
  '''
  return x.one_hot(num_classes)

def clamp(x: Tensor, min_val, max_val) -> Tensor:
  '''
  Clamp the values of a tensor
  '''
  return x.clamp(min_val, max_val)

def square(x: Tensor) -> Tensor:
  '''
  Compute the square of a tensor
  '''
  return x.square()

def abs(x: Tensor) -> Tensor:
  '''
  Compute the absolute value of a tensor
  '''
  return x.abs()

def exp(x: Tensor) -> Tensor:
  '''
  Compute the exponential of a tensor
  '''
  return x.exp()

