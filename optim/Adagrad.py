from autograd.engine import Tensor
from typing import List
import numpy as np

class Adagrad:
  '''
  The Adagrad optimizer inspired by the PyTorch api
  '''
  def __init__(self, params: List[Tensor], lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
    '''
    Initialize the optimizer with the parameters and the learning rate
    params:
    params: list of parameters to optimize or dicts defining parameter groups
    lr: learning rate (default: 1e-2)
    lr_decay: learning rate decay (default: 0)
    weight_decay: weight decay (L2 penalty) (default: 0)
    eps: term added to the denominator to improve numerical stability (default: 1e-10)
    '''
    self.params = params
    self.lr = lr
    self.lr_decay = lr_decay
    self.weight_decay = weight_decay
    self.eps = eps
    self.accumulator_value = [np.full_like(param.data, initial_accumulator_value) for param in self.params]

  
  def step(self):
    '''
    Update the parameters using the gradients
    '''
    for i, param in enumerate(self.params):
      if self.weight_decay != 0:
        param.grad += self.weight_decay * param.data
      # update the accumulator value
      self.accumulator_value[i] += param.grad**2
      # update the parameters
      param.data -= self.lr * param.grad / (np.sqrt(self.accumulator_value[i]) + self.eps)
    # update the learning rate
      self.lr *= 1 / (1 + self.lr_decay)


  def zero_grad(self):
    '''
    Zero the gradients of the parameters
    '''
    for param in self.params:
      param._zero_grad()



