from core import KTorch
from nn.Module import Module
from autograd import Tensor

class CrossEntropyLoss(Module):
  '''
  CrossEntropyLoss

  A class that represents the cross entropy loss
  '''

  def __init__(self, reduction: str = 'mean', label_smoothing: float = 0.0):
    '''
    Initialize the cross entropy loss

    reduction: str - The reduction method
    label_smoothing: float - The label smoothing value, prevents overfitting by adding noise to the labels
    '''
    super().__init__()
    self.reduction = reduction
    self.label_smoothing = label_smoothing

  def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
    '''
    Forward pass

    y_pred: Tensor - The predicted values, the input should be unnormalized logits
    y_true: Tensor - The true values
    '''
    # Compute the softmax
    y_pred = KTorch.softmax(y_pred, axis=-1)

    # Compute the one hot encoding
    num_classes = y_pred.max()
    y_true = KTorch.one_hot(y_true, num_classes + 1)

    # Apply label smoothing if necessary
    if self.label_smoothing > 0:
      y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / y_true.shape[-1]

    # Compute the cross entropy loss
    loss = -y_true * KTorch.log(y_pred)

    # Compute the reduction
    if self.reduction == 'mean':
      return KTorch.mean(loss)
    elif self.reduction == 'sum':
      return KTorch.sum(loss)

  def parameters(self):
    '''
    Return the parameters
    '''
    return []