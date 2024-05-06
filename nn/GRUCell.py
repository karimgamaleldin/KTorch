from autograd.engine import Tensor
from core import KTorch
from nn.Module import Module
from nn.init import simpleUniformInitialization

class GRUCell(Module):
  '''
  GRUCell

  A class that represents a single GRU cell
  '''

  def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
    '''
    Initialize the GRUCell
    '''
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

    # Initialize the weight matrices
    self.w_ir = simpleUniformInitialization((hidden_size, input_size)).transpose(0, 1) # to avoid transposing each time in the forward pass
    self.w_iz = simpleUniformInitialization((hidden_size, input_size)).transpose(0, 1)
    self.w_in = simpleUniformInitialization((hidden_size, input_size)).transpose(0, 1)
    self.w_hr = simpleUniformInitialization((hidden_size, hidden_size))
    self.w_hz = simpleUniformInitialization((hidden_size, hidden_size))
    self.w_hn = simpleUniformInitialization((hidden_size, hidden_size))

    # Initialize the biases
    if bias:
      self.b_ir = simpleUniformInitialization((1, hidden_size)).transpose(0, 1)
      self.b_iz = simpleUniformInitialization((1, hidden_size)).transpose(0, 1)
      self.b_in = simpleUniformInitialization((1, hidden_size)).transpose(0, 1)
      self.b_hr = simpleUniformInitialization((1, hidden_size)).transpose(0, 1)
      self.b_hz = simpleUniformInitialization((1, hidden_size)).transpose(0, 1)
      self.b_hn = simpleUniformInitialization((1, hidden_size)).transpose(0, 1)

  def forward(self, x: Tensor, h: Tensor) -> Tensor:
    '''
    Forward pass
    '''
    r = KTorch.matmul(x, self.w_ir) + KTorch.matmul(h, self.w_hr) + self.b_ir + self.b_hr if self.bias else KTorch.matmul(x, self.w_ir) + KTorch.matmul(h, self.w_hr)
    r = r.sigmoid()
    z = KTorch.matmul(x, self.w_iz) + KTorch.matmul(h, self.w_hz) + self.b_iz + self.b_hz if self.bias else KTorch.matmul(x, self.w_iz) + KTorch.matmul(h, self.w_hz)
    z = z.sigmoid()
    n = KTorch.matmul(x, self.w_in) + r * (KTorch.matmul(h, self.w_hn) + self.b_hn) + self.b_in if self.bias else KTorch.matmul(x, self.w_in) + r * KTorch.matmul(h, self.w_hn)
    n = n.tanh()
    h = (1 - z) * n + z * h
    return h
  
  def parameters(self):
    if self.bias:
      return [self.w_ir, self.w_iz, self.w_in, self.w_hr, self.w_hz, self.w_hn, self.b_ir, self.b_iz, self.b_in, self.b_hr, self.b_hz, self.b_hn]
    return [self.w_ir, self.w_iz, self.w_in, self.w_hr, self.w_hz, self.w_hn]
