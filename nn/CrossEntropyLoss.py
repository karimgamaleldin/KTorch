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
    assert reduction in ['mean', 'sum'], "reduction must be either 'mean' or 'sum'"
    self.reduction = reduction
    self.label_smoothing = label_smoothing

  def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
    '''
    Forward pass

    y_pred: Tensor - The predicted values, the input should be unnormalized logits
    y_true: Tensor - The true values, the input should be the class labels
    '''
    # Compute the softmax
    y_pred = KTorch.softmax(y_pred, axis=-1)

    # Compute the one hot encoding
    num_classes = y_pred.shape[-1]
    y_true_one = KTorch.one_hot(y_true, num_classes)

    # Apply label smoothing if necessary
    if self.label_smoothing > 0:
      y_true_one = y_true_one * (1 - self.label_smoothing) + self.label_smoothing / num_classes # Takes a part of the true label and adds a part of the uniform distribution.
      '''
      Label smoothing is a regularization technique that prevents the model from becoming too confident in its predictions.

      Advantages:
      - Reduces overconfidence, which can lead to overfitting
      - Improves generalization
      - Reduce the impact of noisy labels
      - Improves calibration
      - Reduce sensitivity to class imbalance
      ''' 

    # Compute the cross entropy loss
    loss = -y_true_one * KTorch.log(y_pred)

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