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
    super().__init__('GRU', input_size, hidden_size, num_layers, bias, dropout, bidirectional)
    self.layers = [ GRUCell(input_size if i == 0 else hidden_size, hidden_size, bias) for i in range(num_layers) ]
    self.dropout_layers = [ Dropout(dropout) for _ in range(num_layers) ]
    self.bi_layers = [ GRUCell(input_size if i == 0 else hidden_size, hidden_size, bias) for i in range(num_layers) ] if bidirectional else None

    # Methods for the GRU class are defined in RNNBase.py as they are shared with LSTM and RNN