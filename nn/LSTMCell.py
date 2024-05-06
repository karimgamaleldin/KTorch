from autograd.engine import Tensor
from core import KTorch
from nn.Module import Module
from nn.init import simpleUniformInitialization

class LSTMCell(Module):
  '''
  A class that represents a single LSTM cell
  '''
  def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
    '''
    Initialize the LSTM cell
    '''
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

    # Initialize the weight matrices
    self.w_ii = simpleUniformInitialization((hidden_size, input_size)).transpose(0, 1)
    self.w_if = simpleUniformInitialization((hidden_size, input_size)).transpose(0, 1)
    self.w_ig = simpleUniformInitialization((hidden_size, input_size)).transpose(0, 1)
    self.w_io = simpleUniformInitialization((hidden_size, input_size)).transpose(0, 1)
    self.w_hi = simpleUniformInitialization((hidden_size, hidden_size))
    self.w_hf = simpleUniformInitialization((hidden_size, hidden_size))
    self.w_hg = simpleUniformInitialization((hidden_size, hidden_size))
    self.w_ho = simpleUniformInitialization((hidden_size, hidden_size))

    # Initialize the biases
    if bias:
      self.b_ii = simpleUniformInitialization((1, hidden_size)).transpose(0, 1)
      self.b_if = simpleUniformInitialization((1, hidden_size)).transpose(0, 1)
      self.b_ig = simpleUniformInitialization((1, hidden_size)).transpose(0, 1)
      self.b_io = simpleUniformInitialization((1, hidden_size)).transpose(0, 1)
      self.b_hi = simpleUniformInitialization((1, hidden_size)).transpose(0, 1)
      self.b_hf = simpleUniformInitialization((1, hidden_size)).transpose(0, 1)
      self.b_hg = simpleUniformInitialization((1, hidden_size)).transpose(0, 1)
      self.b_ho = simpleUniformInitialization((1, hidden_size)).transpose(0, 1)

  def forward(self, x: Tensor, h: Tensor) -> Tensor:
    '''
    Forward pass
    '''
    i = KTorch.matmul(x, self.w_ii) + KTorch.matmul(h, self.w_hi) + self.b_ii + self.b_hi if self.bias else KTorch.matmul(x, self.w_ii) + KTorch.matmul(h, self.w_hi)
    i = i.sigmoid()
    f = KTorch.matmul(x, self.w_if) + KTorch.matmul(h, self.w_hf) + self.b_if + self.b_hf if self.bias else KTorch.matmul(x, self.w_if) + KTorch.matmul(h, self.w_hf)
    f = f.sigmoid()
    g = KTorch.matmul(x, self.w_ig) + KTorch.matmul(h, self.w_hg) + self.b_ig + self.b_hg if self.bias else KTorch.matmul(x, self.w_ig) + KTorch.matmul(h, self.w_hg)
    g = g.tanh()
    o = KTorch.matmul(x, self.w_io) + KTorch.matmul(h, self.w_ho) + self.b_io + self.b_ho if self.bias else KTorch.matmul(x, self.w_io) + KTorch.matmul(h, self.w_ho)
    o = o.sigmoid()
    c_dash = f * h + i * g
    h = o * c_dash.tanh()
    return h
  
  def parameters(self):
    if self.bias:
      return [self.w_ii, self.w_if, self.w_ig, self.w_io, self.w_hi, self.w_hf, self.w_hg, self.w_ho, self.b_ii, self.b_if, self.b_ig, self.b_io, self.b_hi, self.b_hf, self.b_hg, self.b_ho]
    return [self.w_ii, self.w_if, self.w_ig, self.w_io, self.w_hi, self.w_hf, self.w_hg, self.w_ho]