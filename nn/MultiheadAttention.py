import nn
from core import KTorch
from autograd.engine import Tensor
from nn import Linear, LayerNorm, Dropout

class MultiheadAttention(nn.Module):
  '''
  A class that represents a multihead attention layer in a neural network.
  '''

  def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, bias: bool=True, add_bias_kv: bool=False, kdim: int=None, vdim: int=None):
    '''
    Initialize the multihead attention layer with the embedding dimension, number of heads, and other parameters
    params:
      embed_dim: int: the embedding dimension
      num_heads: int: the number of heads
      dropout: float: the dropout probability
      bias: bool: whether to include a bias term in the linear layers
      add_bias_kv: bool: whether to include a bias term in the key and value linear layers
      kdim: int: the dimension of the key vectors
      vdim: int: the dimension of the value vectors
    '''
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.dropout = Dropout(dropout)
    self.bias = bias
    self.add_bias_kv = add_bias_kv
    self.kdim = kdim if kdim is not None else embed_dim
    self.vdim = vdim if vdim is not None else embed_dim
    self.head_dim = embed_dim // num_heads
    assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

    self.q = Linear(embed_dim, embed_dim)
    self.k = Linear(embed_dim, self.kdim, bias = self.add_bias_kv)
    self.v = Linear(embed_dim, self.vdim, bias = self.add_bias_kv)

    


  def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor=None, padding_mask: Tensor=None):
    '''
    Forward pass of the multihead attention layer
    '''
    q_matrix, k_matrix, v_matrix = self.q(query), self.k(key), self.v(value)
    q_matrix = q_matrix.view(q_matrix.shape[0], q_matrix.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
    k_matrix = k_matrix.view(k_matrix.shape[0], k_matrix.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
    v_matrix = v_matrix.view(v_matrix.shape[0], v_matrix.shape[1], self.num_heads, self.head_dim)
    attn_output: Tensor = KTorch.matmul(q_matrix, k_matrix.transpose(2, 3))
    attn_output = attn_output / (self.head_dim ** 0.5)
    if attn_mask is not None:
      attn_output = attn_output.masked_fill(attn_mask, float('-inf'))
    if padding_mask is not None:
      attn_output = attn_output.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
    attn_output = KTorch.softmax(attn_output, dim=-1)
    attn_output = self.dropout(attn_output)
    attn_output = KTorch.matmul(attn_output, v_matrix)
    attn_output = attn_output.transpose(1, 2).view(attn_output.shape[0], attn_output.shape[1], self.embed_dim)
    return attn_output

  def parameters(self):
    '''
    Return the parameters of the multihead attention layer
    '''
    return self.qkv.parameters() + self.kv.parameters() + self.attn.parameters() if self.attn is not None else self.qkv.parameters() + self.kv.parameters()