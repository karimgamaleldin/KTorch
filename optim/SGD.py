from autograd.engine import Tensor
from typing import List
import numpy as np

class SGD:
  '''
  The SGD optimizer inspired by the PyTorch api
  '''
  def __init__(self, params: List[Tensor], lr=1e-3, momentum=0.0, dampening=0.0, weight_decay=0, nesterov=False):
    '''
    Initialize the optimizer with the parameters and the learning rate
    params:
      params: list: the parameters of the model
      lr: float: the learning rate of the optimizer
      momentum: float: the momentum coefficient
      dampening: float: the dampening for momentum
      weight_decay: float: the weight decay (L2 penalty)
      nesterov: bool: enables Nesterov momentum
    '''
    self.params = params
    self.lr = lr
    self.prev_velocities = [np.zeros_like(param.data) for param in self.params] # momentum coefficients
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.dampening = dampening
    self.nesterov = nesterov

  def step(self):
    '''
    Update the parameters using the gradients
    '''
    for i, param in enumerate(self.params):
      # Temp value for the gradient, to not change the original gradient
      g = param.grad
      # Apply weight decay
      if self.weight_decay != 0:
        g += self.weight_decay * param.data
        
      # Apply momentum
      if self.momentum != 0:
        self.prev_velocities[i] = self.momentum * self.prev_velocities[i] + (1 - self.dampening) * g

        # Apply Nesterov momentum
        if self.nesterov:
          g += self.momentum * self.prev_velocities[i]
        else: 
          g = self.prev_velocities[i]
      
      # Update the parameters
      param.data -= self.lr * g

  def zero_grad(self):
    '''
    Zero the gradients of the parameters
    '''
    for param in self.params:
      param._zero_grad()
    