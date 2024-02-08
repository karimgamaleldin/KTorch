from autograd.Tensor import Tensor
from typing import List

class SGD:
  def __init__(self, params: List[Tensor], lr=1e-3):
    '''
    Initialize the optimizer with the parameters and the learning rate
    params:
      params: list: the parameters of the model
      lr: float: the learning rate of the optimizer
    '''
    self.params = params
    self.lr = lr

  def step(self):
    '''
    Update the parameters using the gradients
    '''
    for param in self.params:
      param.data -= self.lr * param.grad

  def zero_grad(self):
    '''
    Zero the gradients of the parameters
    '''
    for param in self.params:
      param._zero_grad()
