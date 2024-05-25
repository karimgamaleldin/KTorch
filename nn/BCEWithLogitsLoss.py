from core import KTorch
from autograd.engine import Tensor
import numpy as np
from nn.Module import Module
from nn.BCELoss import BCELoss
from nn.Sigmoid import Sigmoid

class BCEWithLogitsLoss(Module):
    '''
    BCEWithLogitsLoss
    
    A class that represents the binary cross entropy loss with logits, which uses the log-exp-sum trick to avoid numerical instability.
    '''
    def __init__(self, reduction: str = 'mean'):
        '''
        Initialize the binary cross entropy loss with logits
        '''
        super().__init__()
        assert reduction in ['mean', 'sum'], 'reduction must be either mean or sum'
        self.reduction = reduction

    def forward(self, logits: Tensor, y_true: Tensor) -> Tensor:
        '''
        Forward pass
        
        logits: Tensor - The predicted values (logits)
        y_true: Tensor - The true values (class labels)
        '''
        max_val = KTorch.maximum(logits, 0)
        log_exp = KTorch.log(1 + KTorch.exp(-KTorch.abs(logits)))
        bce = max_val + log_exp - logits * y_true 

        if self.reduction == 'mean':
            return KTorch.mean(bce)
        else:
            return KTorch.sum(bce)
    
    def parameters(self):
        return self.bce.parameters() + self.sigmoid.parameters()
