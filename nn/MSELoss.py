from core import KTorch
from autograd.engine import Tensor
from nn.Module import Module


class MSELoss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        loss = KTorch.square(y_true - y_pred)
        if self.reduction == "mean":
            return KTorch.mean(loss)
        elif self.reduction == "sum":
            return KTorch.sum(loss)

    def parameters(self):
        return []
