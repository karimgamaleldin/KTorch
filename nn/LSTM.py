from autograd.engine import Tensor
from core import KTorch
from nn.RNNBase import RNNBase
from nn.LSTMCell import LSTMCell
from nn.Dropout import Dropout

class LSTM(RNNBase):
  '''
  A class that represents an LSTM
  '''

  