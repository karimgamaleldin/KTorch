from core import KTorch
from autograd.engine import Tensor
import numpy as np
from nn.Module import Module

class BCELoss(Module):
  '''
  BCELoss
  
  A class that represents the binary cross entropy loss
  '''

  def __init__(self, reduction: str = 'mean'):
    '''
    Initialize the binary cross entropy loss
    '''
    super().__init__()
    self.reduction = reduction

  def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
    '''
    Forward pass

    y_pred: Tensor - The predicted values
    y_true: Tensor - The true values
    '''
    # Compute the log of the predicted values and clamp the values to -100 like PyTorch
    log_1 = KTorch.log(y_pred)
    log_2 = KTorch.log(1 - y_pred)
    log_1 = KTorch.clamp(log_1, -100, float('inf'))
    log_2 = KTorch.clamp(log_2, -100, float('inf'))

    # Compute the binary cross entropy loss
    loss_term_1 = -y_true * log_1
    loss_term_2 = -(1 - y_true) * log_2
    loss = loss_term_1 + loss_term_2

    # Compute the reduction
    if self.reduction == 'mean':
      return KTorch.mean(loss)
    elif self.reduction == 'sum':
      return KTorch.sum(loss) 

  def parameters(self):
    return []
