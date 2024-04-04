from core import KTorch
from autograd.engine import Tensor
import numpy as np
from nn.Module import Module
from nn.Dropout import Dropout
class PositionalEncoding(Module):
  '''
  A class that represents the positional encoding layer in a transformer model.

  It adds positional information to the input embeddings, using sine and cosine functions.
  PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

  where pos is the position and i is the dimension index.
  '''

  def __init__(self, d_model: int, dropout: float=0.1, max_len: int=1000):
    '''
    Initialize the positional encoding layer with the model dimension, dropout probability, and maximum length.

    Parameters:
      - d_model: int: the model dimension
      - dropout: float: the dropout probability
      - max_len: int: the maximum length of the input sequences
    '''
    super().__init__()
    self.d_model = d_model
    # self.dropout = nn.Dropout(dropout)
    self.max_len = max_len
    self.dropout = Dropout(dropout)
    self._generate_positional_encoding()

  def _generate_positional_encoding(self):
    # Create a tensor of zeros, to store the positional encoding
    pe = np.zeros((self.max_len, self.d_model))
    # Create the power of 10000
    div = 2 * np.arange(self.d_model // 2, dtype=np.float32) / self.d_model
    div = np.power(10000, div)
    # Create the position tensor
    pos = np.arange(self.max_len, dtype=np.float32).reshape(-1, 1)
    # Compute the positional encoding
    pe[:, 0::2] = np.sin(pos / div)
    pe[:, 1::2] = np.cos(pos / div)
    # So the first dimension is the batch dimension
    self.pe = np.expand_dims(pe, 0)

  def forward(self, x: Tensor) -> Tensor:
    ten_pe = KTorch.Tensor(self.pe[:, :x.shape[1], :])
    x = x + ten_pe
    return self.dropout(x)

  def parameters(self):
    '''
    Return the parameters of the positional encoding layer
    '''
    return []

