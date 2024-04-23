import nn
from core import KTorch
from autograd.engine import Tensor
from nn import Linear, LayerNorm, Dropout, MultiheadAttention, ReLU, GELU

class TransformerDecoderLayer(nn.Module):
  def __init__(self, d_model: int, nhead: int, dim_feedforward: int=2048, dropout: float=0.1, activation: str='relu', bias: bool=True):
    self.d_model = d_model
    self.nhead = nhead
    self.dim_feedforward = dim_feedforward
    self.dropout = Dropout(dropout)
    self.activation = activation
    self.bias = bias


    self.mha = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias)
    self.norm1 = LayerNorm(d_model)
    activation = ReLU if activation == 'relu' else GELU
    self.ffn = nn.Sequential(
      Linear(d_model, dim_feedforward),
      activation(),
      Linear(dim_feedforward, d_model)
    )
    self.norm2 = LayerNorm(d_model)

  def forward(self, tgt: Tensor, memory: Tensor, attn_mask: Tensor=None, padding_mask: Tensor=None):
    inp = self.mha(tgt, tgt, tgt, attn_mask=attn_mask, padding_mask=padding_mask)
    inp = self.dropout(inp)
    inp = tgt + inp
    inp = self.norm1(inp)
    inp2 = self.ffn(inp)
    inp2 = self.dropout(inp2)
    inp2 = inp + inp2
    inp2 = self.norm2(inp2)
    return inp2 

  def parameters(self):
    pass

