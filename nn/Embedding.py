from autograd.engine import Tensor
from core import KTorch
from nn.Module import Module


class Embedding(Module):
  def __init__(self, num_embeddings: int, embedding_dim: int):
    '''
    Initialize the Embedding
    '''
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.weight = KTorch.randn((num_embeddings, embedding_dim))