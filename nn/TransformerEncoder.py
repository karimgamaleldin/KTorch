import nn
from core import KTorch
from autograd.engine import Tensor
from nn import TransformerEncoderLayer, Sequential

class TransformerEncoder(nn.Module):
  '''
  A class that represents a multi-layered transformer encoder in a neural network.
  '''
  def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int):
    self.encoder_layer = encoder_layer
    self.num_layers = num_layers

    self.layers = Sequential(*[encoder_layer for _ in range(num_layers)])

  def forward(self, x: Tensor, attn_mask: Tensor=None, padding_mask: Tensor=None):
    for layer in self.layers:
      x = layer(x, attn_mask=attn_mask, padding_mask=padding_mask)
    return x
  
  def parameters(self):
    return self.layers.parameters()