from autograd.engine import Tensor
from core import KTorch
from nn.Module import Module


class Embedding(Module):
  def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None, max_norm: float = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False):
    '''
    Initialize the Embedding
    '''
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.padding_idx = padding_idx
    self.max_norm = max_norm
    self.norm_type = norm_type
    self.scale_grad_by_freq = scale_grad_by_freq
    
    self.weight = Tensor.randn(num_embeddings, embedding_dim)
