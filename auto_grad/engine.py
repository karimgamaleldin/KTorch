import numpy as np

class Parameter:
  '''
  Stores a numpy array and its gradient
  '''

  def __init__(self, data):
    '''
    Initialize the parameter with the data
    params:
      data: numpy array: the data of the parameter
    '''
    self.data = data
    self.grad = np.zeros_like(data)

  def __add__(self, other):
    '''
    Add the data of the parameter with another parameter
    '''
    out = Parameter(self.data + other.data)
    return out
  
  def __sub__(self, other):
    '''
    Subtract the data of the parameter with another parameter
    '''
    out = Parameter(self.data - other.data)
    return out
  
  def __mul__(self, other):
    '''
    Multiply the data of the parameter with another parameter
    '''
    out = Parameter(self.data * other.data)
    return out
  
  def __pow__(self, other):
    '''
    Raise the data of the parameter to the power of a number
    '''
    assert isinstance(other, (int, float)), "The exponent must be a number"
    out = Parameter(self.data ** other)
    return out
  
  def __repr__(self):
    return f"Parameter: {self.data} , Gradient: {self.grad}"
  
  
  

  
