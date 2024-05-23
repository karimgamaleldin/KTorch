from optim.Optim import Optim
from autograd.engine import Tensor
from typing import List
import numpy as np

class Adam(Optim):
  '''
  The Adam & AMSGrad optimizers inspired by the PyTorch api

  AMSGrad is a variant of Adam that uses the maximum of past squared gradients
    - It helps in maintaining the learning rate and the convergence of the model
  '''

  def __init__(self, params: List[Tensor], lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, amsgrad=False):
    '''
    Initialize the optimizer with the parameters and the learning rate
    params:
      params: list: the parameters of the model
      lr: float: the learning rate of the optimizer
      betas: tuple: the coefficients used for computing running averages of gradient and its square
      eps: float: a term added to the denominator to improve numerical stability
      weight_decay: float: the weight decay (L2 penalty)
      amsgrad: bool: if True, use the AMSGrad variant of this algorithm
    '''
    super().__init__(params)
    self.lr = lr
    self.betas = betas
    self.eps = eps
    self.weight_decay = weight_decay
    self.amsgrad = amsgrad
    self.v = [np.zeros_like(param.data) for param in self.params]
    self.s = [np.zeros_like(param.data) for param in self.params] 
    self.s_max = [np.zeros_like(param.data) for param in self.params] # maximum of the v
    self.t = 0


  def step(self):
    '''
    Update the parameters of the model
    '''
    self.t += 1
    for i, param in enumerate(self.params):
      # Temporary variable to store the gradient
      g = param.grad
      # Apply weight decay
      if self.weight_decay != 0:
        g += self.weight_decay * param.data
      # Update the momentum
      self.v[i] = self.betas[0] * self.v[i] + (1 - self.betas[0]) * g
      # Update the second moment
      self.s[i] = self.betas[1] * self.s[i] + (1 - self.betas[1]) * g ** 2
      # Compute the bias-corrected first moment
      v_hat = self.v[i] / (1 - self.betas[0] ** self.t)
      # Compute the bias-corrected second
      s_hat = self.s[i] / (1 - self.betas[1] ** self.t)
      # Update the parameters
      if self.amsgrad:
        self.s_max[i] = np.maximum(s_hat, self.s_max[i])
        param.data -= self.lr * (v_hat / (np.sqrt(self.s_max[i]) + self.eps))
      else:
        param.data -= self.lr * (v_hat / (np.sqrt(s_hat) + self.eps))
