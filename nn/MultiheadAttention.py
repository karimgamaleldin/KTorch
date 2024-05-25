from core import KTorch
from autograd.engine import Tensor
from nn import Linear, Dropout, Module


class MultiheadAttention(Module):
    """
    A class that represents a multihead attention layer in a neural network.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        kdim: int = None,
        vdim: int = None,
    ):
        """
        Initialize the multihead attention layer with the embedding dimension, number of heads, and other parameters
        params:
          embed_dim: int: the embedding dimension
          num_heads: int: the number of heads
          dropout: float: the dropout probability
          bias: bool: whether to include a bias term in the linear layers
          add_bias_kv: bool: whether to include a bias term in the key and value linear layers
          kdim: int: the dimension of the key vectors
          vdim: int: the dimension of the value vectors
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = Dropout(dropout)
        self.bias = bias
        self.add_bias_kv = add_bias_kv
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.head_dim = embed_dim // num_heads
        assert (
            self.embed_dim % self.num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.q = Linear(embed_dim, embed_dim, bias=bias)
        self.k = Linear(embed_dim, self.kdim, bias=self.add_bias_kv)
        self.v = Linear(embed_dim, self.vdim, bias=self.add_bias_kv)

        self.out = Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor = None,
        padding_mask: Tensor = None,
    ):
        """
        Forward pass of the multihead attention layer
        """
        batch_size = query.shape[0]
        q_matrix, k_matrix, v_matrix = (
            self.q(query),
            self.k(key),
            self.v(value),
        )  # Get the query, key, and value matrices
        q_matrix = q_matrix.view(
            q_matrix.shape[0], q_matrix.shape[1], self.num_heads, self.head_dim
        ).transpose(
            1, 2
        )  # Reshape the query matrix
        k_matrix = k_matrix.view(
            k_matrix.shape[0], k_matrix.shape[1], self.num_heads, self.head_dim
        ).transpose(
            1, 2
        )  # Reshape the key matrix
        v_matrix = v_matrix.view(
            v_matrix.shape[0], v_matrix.shape[1], self.num_heads, self.head_dim
        ).transpose(
            1, 2
        )  # Reshape the value matrix

        attn_scores: Tensor = KTorch.matmul(
            q_matrix, k_matrix.transpose(-2, -1)
        )  # Compute the attention scores
        attn_scores = attn_scores / (self.head_dim**0.5)  # Scale the attention scores

        # Masking
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))
        if padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = KTorch.softmax(
            attn_scores, axis=-1
        )  # Apply the softmax function
        attn_weights, _ = self.dropout(attn_weights)  # Apply dropout

        attn_output = KTorch.matmul(attn_weights, v_matrix)
        attn_output = attn_output.transpose(1, 2).view(
            batch_size, -1, self.embed_dim
        )  # Reshape the output
        output = self.out(
            attn_output
        )  # Apply the output linear layer, which combines the heads
        return output

    def parameters(self):
        """
        Return the parameters of the multihead attention layer
        """
        return (
            self.q.parameters()
            + self.k.parameters()
            + self.v.parameters()
            + self.out.parameters()
            + self.dropout.parameters()
        )
