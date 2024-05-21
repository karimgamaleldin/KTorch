from core import KTorch
from autograd.engine import Tensor
import numpy as np
from nn.Module import Module
from nn.BCELoss import BCELoss
from nn.Sigmoid import Sigmoid

class BCEWithLogitsLoss(Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.sigmoid = Sigmoid()
        self.bce = BCELoss(reduction)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        y_pred = self.sigmoid(y_pred)
        return self.bce(y_pred, y_true)
    
    def parameters(self):
        return self.bce.parameters() + self.sigmoid.parameters()
