from autograd.engine import Tensor
from core import KTorch
from nn.Module import Module

class RNNBase(Module):
  '''
  A class that represents a base class for RNNs (RNN, LSTM, GRU)
  '''
  
  def __init__(self, input_size: int, hidden_size: int, num_layers: int, bias: bool = True, dropout: float = 0.0, bidirectional: bool = False):
    '''
    Initialize the base RNN class
    '''
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bias = bias
    self.dropout = dropout
    self.bidirectional = bidirectional
    self.hidden = None
    self.cell = None
    self.dropout_layer = None

  def init_hidden(self, batch_size: int) -> Tensor:
    '''
    Initialize the hidden state
    '''
    raise NotImplementedError
  
  def init_cell(self, batch_size: int) -> Tensor:
    '''
    Initialize the cell state
    '''
    raise NotImplementedError
  
  def forward(self, x: Tensor) -> Tensor:
    '''
    Perform the forward pass of the RNN
    '''
    raise NotImplementedError
  