import nn
from core import KTorch
from autograd.engine import Tensor
from nn import Linear, LayerNorm, Dropout, MultiheadAttention, LayerNorm, ReLU, GELU


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        bias: bool = True,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = Dropout(dropout)
        self.activation = activation
        self.bias = bias

        self.mha = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias)
        self.norm1 = LayerNorm(d_model)
        activation = ReLU if activation == "relu" else GELU
        self.ffn = nn.Sequential(
            Linear(d_model, dim_feedforward),
            activation(),
            Linear(dim_feedforward, d_model),
        )
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, tgt, attn_mask=None, padding_mask=None):
        inp = self.mha(tgt, tgt, tgt, attn_mask=attn_mask, padding_mask=padding_mask)
        inp = self.dropout(inp)
        inp = tgt + inp
        inp = self.norm1(inp)
        inp2 = self.cross_attn(
            inp, tgt, tgt, attn_mask=attn_mask, padding_mask=padding_mask
        )
        inp2 = self.dropout(inp2)
        inp2 = inp + inp2
        inp2 = self.norm2(inp2)
        inp3 = self.ffn(inp2)
        inp3 = self.dropout(inp3)
        inp3 = inp2 + inp3
        inp3 = self.norm3(inp3)
        return inp3

    def parameters(self):
        return (
            self.mha.parameters()
            + self.cross_attn.parameters()
            + self.ffn.parameters()
            + self.norm1.parameters()
            + self.norm2.parameters()
            + self.norm3.parameters()
        )
