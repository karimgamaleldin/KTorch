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

    def forward(self, logits: Tensor, y_true: Tensor) -> Tensor:
        t_n = KTorch.clamp(-logits, 0, float('inf')) # To avoid taking exponential of positive numbers

        log_1_plus_exp_neg_abs = KTorch.log(1 + KTorch.exp(-KTorch.abs(-t_n - logits))) + t_n

    
    def parameters(self):
        return self.bce.parameters() + self.sigmoid.parameters()
