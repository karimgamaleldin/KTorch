from autograd.engine import Tensor
from core import KTorch
from nn.RNNBase import RNNBase
from nn.GRUCell import GRUCell
from nn.Dropout import Dropout

class GRU(RNNBase):
  '''
  A class that represents a GRU
  '''

  def __init__(self, input_size: int, hidden_size: int, num_layers: int, bias: bool = True, dropout: float = 0.0, bidirectional: bool = False):
    '''
    Initialize the GRU
    '''
    super().__init__(input_size, hidden_size, num_layers, bias, dropout, bidirectional)
    self.gru_layers = [ GRUCell(input_size, hidden_size, bias) for _ in range(num_layers) ]
    self.dropout_layers = [ Dropout(dropout) for _ in range(num_layers) ]
    self.bidirectional = bidirectional
    self.bi_rnn_layers = [ GRUCell(input_size, hidden_size, bias) for _ in range(num_layers) ] if bidirectional else None

  def forward(self, x: Tensor, h_0: Tensor=None, h_0_back: Tensor=None) -> Tensor:
    '''
    Perform the forward pass of the GRU
    '''
    if self.bidirectional:
      out = self._forward_bidirectional(x, h_0)
    else:
      out = self._forward_single_direction(x, h_0, h_0_back)

    return out
  
  def _forward_bidirectional(self, x: Tensor, h_0: Tensor, h_0_back: Tensor) -> Tensor:
    '''
    Perform the forward pass of the bidirectional GRU
    '''
    forward_out, forward_hidden = self._forward_single_direction(x, h_0)
    backward_out, backward_hidden = self._forward_single_direction(x.flip(1), h_0_back)
    out = [ KTorch.cat((forward, backward), axis=-1) for forward, backward in zip(forward_out, backward_out) ]
    hidden = [ KTorch.cat((forward, backward), axis=-1) for forward, backward in zip(forward_hidden, backward_hidden) ]
    return out, hidden
  
  def _forward_single_direction(self, x: Tensor, h_0: Tensor) -> Tensor:
    '''
    Perform the forward pass of the single direction GRU
    '''
    batch_size, seq_len, _ = x.shape
    if h_0 is None:
      h_0 = [ KTorch.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers) ]

    h_prev = h_0
    h_curr = []
    output = []
    for i in range(seq_len): # iterate over the sequence length
      for j in range(self.num_layers):
        temp = self.gru_layers[j](x[:, i, :], h_prev[j])
        temp = self.dropout_layers[j](h_prev[j])
        h_curr.append(temp)
      output.append(h_curr[-1])
      h_prev = h_curr
      h_curr = []

    return output, h_prev