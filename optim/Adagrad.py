from optim.Optim import Optim
from autograd.engine import Tensor
from typing import List
import numpy as np

class Adagrad(Optim):
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
    super().__init__(params)
    self.lr = lr
    self.initial_lr = lr
    self.lr_decay = lr_decay
    self.weight_decay = weight_decay
    self.eps = eps
    self.accumulator_value = [np.full_like(param.data, initial_accumulator_value) for param in self.params]
    self.step_count = 0

  def step(self):
    '''
    Update the parameters using the gradients
    '''
    self.step_count += 1

    if self.lr_decay != 0:
      self.lr = self.initial_lr / (1 + self.lr_decay * self.step_count)

    for i, param in enumerate(self.params):
      # Temp value for the gradient, to not change the original gradient
      g = param.grad
      # add the weight decay
      if self.weight_decay != 0:
        g += self.weight_decay * param.data # L2 penalty
      # update the accumulator value
      self.accumulator_value[i] += g**2
      # update the parameters
      param.data -= self.lr * g / (np.sqrt(self.accumulator_value[i]) + self.eps)
    # update the learning rate

