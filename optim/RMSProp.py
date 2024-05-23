from optim.Optim import Optim
from autograd.engine import Tensor
from typing import List
import numpy as np

class RMSProp(Optim):
  '''
  The RMSProp & centered RMSProp optimizers inspired by the PyTorch api

  centered RMSProp:
    - A variant of RMSProp that uses centered gradients to improve the stability of the optimization algorithm
    - We normalize the gradient by the variance of the gradients instead of the root mean square of the gradients
  '''
  def __init__(self, params: List[Tensor], lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
    '''
    Initialize the optimizer with the parameters and the learning rate
    params:
      params: list: the parameters of the model
      lr: float: the learning rate of the optimizer
      alpha: float: the smoothing constant
      eps: float: a term added to the denominator to improve numerical stability
      weight_decay: float: the weight decay (L2 penalty)
      momentum: float: the momentum coefficient
      centered: bool: if True, compute the centered RMSProp
    '''
    super().__init__(params)
    self.lr = lr
    self.alpha = alpha
    self.eps = eps
    self.weight_decay = weight_decay
    self.momentum = momentum
    self.centered = centered
    self.accumulator_value = [np.zeros_like(param.data) for param in self.params] # store the moving average of the squared gradient to scale the learning rate
    self.prev_velocities = [np.zeros_like(param.data) for param in self.params] # store the momentum coefficients for each parameter to apply momentum to the optimizer
    self.centered_accumulator = None  
    if self.centered:
      self.centered_accumulator = [np.zeros_like(param.data) for param in self.params] # store the moving average of the gradient to compute the centered RMSProp, make the optimizer more stable and helps in taking bigger steps in steep regions
    

  def step(self):
    '''
    Update the parameters using the gradients
    '''
    for i, param in enumerate(self.params):
      # Temp value for the gradient, to not change the original gradient
      g = param.grad 

      # Apply weight decay
      if self.weight_decay != 0:
        g += self.weight_decay * param.data # L2 penalty

      # update the accumulator value, moving average of the squared gradient
      self.accumulator_value[i] = self.alpha * self.accumulator_value[i] + (1 - self.alpha) * g ** 2  
      v_temp = self.accumulator_value[i]

      # Apply centered RMSProp
      if self.centered:
        self.centered_accumulator[i] = self.alpha * self.centered_accumulator[i] + (1 - self.alpha) * g
        v_temp = v_temp - (self.centered_accumulator[i] ** 2)

      # Apply momentum
      if self.momentum != 0:
        self.prev_velocities[i] = self.momentum * self.prev_velocities[i] + g / (np.sqrt(v_temp) + self.eps) # update the velocity
        param.data -= self.lr * self.prev_velocities[i]
      else :
        param.data -= self.lr * g / (np.sqrt(v_temp) + self.eps)

