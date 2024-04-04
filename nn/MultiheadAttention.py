import nn
from core import KTorch
from autograd.engine import Tensor
from nn import Linear, LayerNorm, Dropout

class MultiheadAttention(nn.Module):
  '''
  A class that represents a multihead attention layer in a neural network.
  '''

  def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, bias: bool=True, add_bias_kv: bool=False, add_zero_attn: bool=False, kdim: int=None, vdim: int=None):
    '''
    Initialize the multihead attention layer with the embedding dimension, number of heads, and other parameters
    params:
      embed_dim: int: the embedding dimension
      num_heads: int: the number of heads
      dropout: float: the dropout probability
      bias: bool: whether to include a bias term in the linear layers
      add_bias_kv: bool: whether to include a bias term in the key and value linear layers
      add_zero_attn: bool: whether to add a zero attention vector
      kdim: int: the dimension of the key vectors
      vdim: int: the dimension of the value vectors
    '''
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.dropout = Dropout(dropout)
    self.bias = bias
    self.add_bias_kv = add_bias_kv
    self.add_zero_attn = add_zero_attn
    self.kdim = kdim if kdim is not None else embed_dim
    self.vdim = vdim if vdim is not None else embed_dim
    self.head_dim = embed_dim // num_heads
    assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

    


  def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Tensor=None, need_weights: bool=True, attn_mask: Tensor=None, average_attn_weights: bool=True, is_causal=False):
    pass 

  def _reset_parameters(self):
    pass

  def parameters(self):
    '''
    Return the parameters of the multihead attention layer
    '''
    return self.qkv.parameters() + self.kv.parameters() + self.attn.parameters() if self.attn is not None else self.qkv.parameters() + self.kv.parameters()