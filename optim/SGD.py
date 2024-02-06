import numpy as np
from auto_grad.engine import Parameter
from typing import List

class SGD:
  def __init__(self, parameters: List[Parameter], lr=1e-3):
    '''
    Initialize the optimizer with the parameters and the learning rate
    params:
      parameters: list: the parameters of the model
      lr: float: the learning rate of the optimizer
    '''
    self.parameters = parameters
    self.lr = lr

  def step(self):
    '''
    Update the parameters using the gradients
    '''
    for param in self.parameters:
      param.data -= self.lr * param.grad

  def zero_grad(self):
    '''
    Zero the gradients of the parameters
    '''
    for param in self.parameters:
      param.grad = np.zeros_like(param.grad)
