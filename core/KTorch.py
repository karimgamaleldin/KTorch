from autograd.Tensor import Tensor
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
  return x.matmul(y)