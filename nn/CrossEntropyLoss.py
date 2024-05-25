from core import KTorch 
from nn.Module import Module
from autograd import Tensor

class CrossEntropyLoss(Module):
  '''
  CrossEntropy
  
  A class that represents the cross entropy loss
  '''

  def __init__(self, reduction: str = 'mean', label_smoothing: float = 0.0):
    '''
    Initialize the cross entropy loss

    reduction: str - The reduction method
    '''
    super().__init__()
    assert reduction in ['mean', 'sum'], "reduction must be either 'mean' or 'sum'"
    self.reduction = reduction
    self.label_smoothing = label_smoothing

  def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
    '''
    Forward pass

    y_pred: Tensor - The predicted values (logits)
    y_true: Tensor - The true values (class labels)
    '''

    # one hot encoding
    num_classes = y_pred.shape[-1]
    y_true_one = KTorch.one_hot(y_true, num_classes)

    # label smoothing
    if self.label_smoothing > 0:
      y_true_one = y_true_one * (1 - self.label_smoothing) + self.label_smoothing / num_classes
      '''
      Label smoothing is a regularization technique that prevents the model from becoming too confident in its predictions.

      Advantages:
      - Reduces overconfidence, which can lead to overfitting
      - Improves generalization
      - Reduce the impact of noisy labels
      - Improves calibration
      - Reduce sensitivity to class imbalance
      ''' 
    
    # Calculate the log probabilities
    max_logit = KTorch.max(y_pred, axis=-1, keepdims=True)
    logits = y_pred - max_logit
    log_sum_exp = KTorch.log(KTorch.sum(KTorch.exp(logits), axis=-1, keepdims=True))
    log_probs = logits - log_sum_exp
    loss = -1 * KTorch.sum(y_true_one * log_probs, axis=-1)

    # Reduction
    if self.reduction == 'mean':
      return KTorch.mean(loss)
    else:
      return KTorch.sum(loss)

  def parameters(self):
    return []
    