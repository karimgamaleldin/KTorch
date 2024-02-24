import numpy as np

class Tensor:
  '''
  Stores a numpy array and its gradient
  '''

  def __init__(self, data, _prev=(), _op='', label=''):
    '''
    Initialize the tensor with the data
    params:
      data: numpy array: the data of the tensor
      _prev: tuple: the previous tensors that were used to compute the current tensor
      op: string: the operation that was used to compute the current tensor
    '''
    self.data = np.array(data, dtype=np.float32)
    self.grad = np.zeros_like(data, dtype=np.float32)
    self._prev = _prev
    self._op = _op
    self.label = label
    self._backward = lambda: None

  def __add__(self, other):
    '''
    Add the data of the tensor with another tensor
    '''
    if isinstance(other, Tensor):
      if self.data.shape != other.data.shape:
        raise ValueError("The shapes of the tensors must be the same")
      out = Tensor(self.data + other.data, _prev=(self, other), _op='add', label=f"{self.label} + {other.label}")
    elif isinstance(other, (int, float)):
      out = Tensor(self.data + other, _prev=(self,), _op='add', label=f"{self.label} + {other}")
    else:
      raise TypeError("The tensor must be a tensor or a number")
    
    def _backward():
      self.grad += out.grad
      if isinstance(other, Tensor):
        other.grad += self.grad

    out._backward = _backward
    return out
  
  def __radd__(self, other):
    '''
    Add the data of the tensor with another tensor or number
    '''
    return self.__add__(other)
  
  def __sub__(self, other):
    '''
    Subtract the data of the tensor with another tensor
    '''
    return self.__add__(-1 * other)
  
  def __rsub__(self, other):
    '''
    Subtract the data of the tensor from another tensor or number
    '''
    return (-1 * self).__add__(other)
  
  def __mul__(self, other):
    '''
    Multiply the data of the tensor with another tensor
    '''
    if isinstance(other, Tensor):
      if self.data.shape != other.data.shape:
        raise ValueError("The shapes of the tensors must be the same")
      out = Tensor(self.data * other.data, _prev=(self, other), _op='mul', label=f"{self.label} * {other.label}")
    elif isinstance(other, (int, float)):
      out = Tensor(self.data * other, _prev=(self,), _op='mul', label=f"{self.label} * {other}")
    else:
      raise TypeError("The tensor must be a tensor or a number")
    
    def _backward():
      self.grad += other.data * out.grad if isinstance(other, Tensor) else other * out.grad
      if isinstance(other, Tensor):
        other.grad += self.data * out.grad

    out._backward = _backward
    return out
  
  def __rmul__(self, other):
    '''
    Multiply the data of the tensor with another tensor or number
    '''
    return self * other 
  
  def __pow__(self, other):
    '''
    Raise the data of the tensor to the power of a number
    '''
    assert isinstance(other, (int, float)), "The exponent must be a number"
    out = Tensor(self.data ** other, _prev=(self,), _op='pow', label=f"{self.label} ** {other}")
    def _backward():
      self.grad += other * self.data ** (other - 1) * out.grad

    out._backward = _backward
    return out
  
  def __repr__(self):
    return f"tensor ({self.label}): {self.data}"
  
  def __str__(self):
    return f"tensor ({self.label}): {self.data}"
  
  def ReLU(self):
    '''
    Apply the ReLU function to the tensor
    '''
    out = Tensor(np.maximum(self.data, 0), _prev=(self,), _op='ReLU', label=f"ReLU({self.label})")
    def _backward():
      self.grad += (self.data > 0) * out.grad

    out._backward = _backward
    return out
  
  def sigmoid(self):
    '''
    Apply the sigmoid function to the tensor
    '''
    t = 1/(1+np.exp(-self.data))
    out = Tensor(t, _prev=(self,), _op='sigmoid', label=f"sigmoid({self.label})")

    def _backward():
      self.grad += t * (1 - t) * out.grad

    out._backward = _backward
    return out
  
  def tanh(self):
    '''
    Apply the tanh function to the tensor
    '''
    t = np.tanh(self.data)
    out = Tensor(t, _prev=(self,), _op='tanh', label=f"tanh({self.label})")
    def _backward():
      self.grad += (1 - t ** 2) * out.grad 

    out._backward = _backward
    return out
  
  def __neg__(self):
    '''
    Negate the data of the tensor
    '''
    return self.__mul__(-1)
  
  def __truediv__(self, other):
    '''
    Divide the data of the tensor by another tensor
    '''
    return self * other ** -1

  def exp(self):
    '''
    Apply the exponential function to the tensor
    '''
    t = np.exp(self.data)
    out = Tensor(t, _prev=(self,), _op='exp', label=f"exp({self.label})")
    def _backward():
      self.grad += t * out.grad

    out._backward = _backward
    return out
  
  def backward(self):
    '''
    Compute the gradient for all the previous tensors using topological sort
    '''
    topo = []
    visited = set()
    def dfs(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          dfs(child)
        topo.append(v) 

    dfs(self)
    self.grad = np.ones_like(self.data)
    for node in reversed(topo):
      node._backward()
  
  def matmul(self, other):
    '''
    Compute the matrix multiplication of 2 tensors
    '''
    
    if self.data.ndim == 1:
      self.data = self.data.reshape(1, -1)
    if other.data.ndim == 1:
      other.data = other.data.reshape(1, -1)
    if self.data.shape[1] != other.data.shape[0]:
      raise ValueError("The shapes of the tensors must be the same")
    t = np.matmul(self.data, other.data)
    out = Tensor(t, _prev=(self, other), _op='matmul', label=f"{self.label} @ {other.label}")
    def _backward():
      '''
      self, self.grad - matrix of shape (n, m)
      other, other.grad - tensor of shape (m, p)
      out, out.grad - tensor of shape (n, p)
      '''
      self.grad += np.matmul(out.grad, other.data.T) # (n, p) * (p, m) = (n, m)
      other.grad += np.matmul(self.data.T, out.grad) # (m, n) * (n, p) = (m, p)
    out._backward = _backward
    return out
  
  def _zero_grad(self):
    '''
    Zero the gradient of the tensor
    '''
    self.grad = np.zeros_like(self.data)

  def flatten(self, start_dim=None, end_dim=-1):
    '''
    Flatten the tensor
    '''
    if start_dim is None:
      start_dim = self.data.shape[0]
    t = self.data.reshape(start_dim, end_dim)
    out = Tensor(t, _prev=(self,), _op='flatten', label=f"flatten({self.label})")
    def _backward():
      self.grad += out.grad.reshape(self.data.shape)

    out._backward = _backward
    return out