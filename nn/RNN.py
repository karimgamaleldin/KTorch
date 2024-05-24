from autograd.engine import Tensor
from core import KTorch
from nn.RNNBase import RNNBase
from nn.RNNCell import RNNCell
from nn.Dropout import Dropout

class RNN(RNNBase):
  '''
  A class that represents a RNN
  '''
  
  def __init__(self, input_size: int, hidden_size: int, num_layers: int, nonlinearity: str = 'tanh', bias: bool = True, dropout: float = 0.0, bidirectional: bool = False):
    '''
    Initialize the RNN

    Parameters:
      input_size (int): The size of the input
      hidden_size (int): The size of the hidden state
      num_layers (int): The number of layers
      nonlinearity (str): The nonlinearity function to use (default is 'tanh')
      bias (bool): Whether to use bias (default is True)
      dropout (float): The dropout probability (default is 0.0)
      bidirectional (bool): Whether to use bidirectional RNN (default is False)
    '''
    super().__init__(input_size, hidden_size, num_layers, bias, dropout, bidirectional)
    self.nonlinearity = nonlinearity
    self.rnn_layers = [RNNCell(input_size, hidden_size, bias, nonlinearity) for _ in range(num_layers)]
    self.dropout_layers = [ Dropout(dropout) for _ in range(num_layers) ]
    self.bidirectional = bidirectional
    self.bi_rnn_layers = [RNNCell(hidden_size, hidden_size, bias, nonlinearity) for _ in range(num_layers)] if bidirectional else None

  def forward(self, x: Tensor, h_0: Tensor=None, h_0_back: Tensor=None) -> Tensor:
    '''
    Perform the forward pass of the RNN
    '''
    if self.bidirectional:
      out, hidden = self._forward_bidirectional(x, h_0, h_0_back)
    else:
      out, hidden = self._forward_single_direction(x, h_0)

    return out, hidden
  
  def _forward_bidirectional(self, x: Tensor, h_0: Tensor, h_0_back: Tensor) -> Tensor:
    '''
    Perform the forward pass of the bidirectional RNN
    '''
    forward_out, forward_hidden = self._forward_single_direction(x, h_0)
    backward_out, backward_hidden = self._forward_single_direction(x.flip(1), h_0_back)
    out = [ KTorch.cat((forward, backward), axis=-1) for forward, backward in zip(forward_out, backward_out[::-1])]
    hidden = [ KTorch.cat((forward, backward), axis=-1) for forward, backward in zip(forward_hidden, backward_hidden) ]
    return out, hidden 

  def _forward_single_direction(self, x: Tensor, h_0: Tensor) -> Tensor:
    '''
    Perform the forward pass of the single direction RNN
    '''
    batch_size, seq_len, _ = x.shape
    if h_0 is None:
      h_0 = [ KTorch.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers) ]

    h_prev = h_0
    h_curr = []
    output = []
    for i in range(seq_len): # iterate over the sequence length
      x_i = x[:, i, :]
      for j in range(self.num_layers): # iterate over the number of layers
        temp = self.rnn_layers[j](x_i, h_prev[j])
        temp, _ = self.dropout_layers[j](temp) if self.training else temp
        h_curr.append(temp) # append the hidden state to the current hidden states
      output.append(h_curr[-1]) # append the last hidden state to the output
      h_prev = h_curr
      h_curr = []
    return output, h_prev # return the output of each time step and the hidden states of the last time step
  
  def parameters(self):
    out = [ param for layer in self.rnn_layers for param in layer.parameters() ] 
    if self.bidirectional:
      out += [ param for layer in self.bi_rnn_layers for param in layer.parameters() ]
    return out

    