from core import KTorch
from nn.Module import Module
from autograd import Tensor

class CrossEntropyLoss(Module):
  '''
  CrossEntropyLoss

  A class that represents the cross entropy loss
  '''

  def __init__(self, weight: Tensor = None, reduction: str = 'mean'):
    '''
    Initialize the cross entropy loss
    '''
    super().__init__()
    self.weight = weight
    self.reduction = reduction

  def forward(self, x: Tensor, y: Tensor) -> Tensor:
    '''
    Forward pass
    '''
    # Compute the softmax
    pass 

  def parameters(self):
    '''
    Return the parameters
    '''
    return []