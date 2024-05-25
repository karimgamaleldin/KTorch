import nn
from core import KTorch
from autograd.engine import Tensor
from nn import TransformerDecoderLayer
from nn import TransformerDecoderLayer, Sequential


class TransformerDecoder(nn.Module):
    """
    A class that represents a multi-layered transformer decoder in a neural network.
    """

    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int):
        self.decoder_layer = decoder_layer
        self.num_layers = num_layers

        self.layers = Sequential(*[decoder_layer for _ in range(num_layers)])

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        attn_mask: Tensor = None,
        padding_mask: Tensor = None,
    ):
        for layer in self.layers:
            x = layer(x, memory, attn_mask=attn_mask, padding_mask=padding_mask)
        return x

    def parameters(self):
        return self.layers.parameters()
