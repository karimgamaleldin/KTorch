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
    self.w_ir = simpleUniformInitialization((input_size, hidden_size)) 
    self.w_iz = simpleUniformInitialization((input_size, hidden_size)) 
    self.w_in = simpleUniformInitialization((input_size, hidden_size)) 
    self.w_hr = simpleUniformInitialization((hidden_size, hidden_size))
    self.w_hz = simpleUniformInitialization((hidden_size, hidden_size))
    self.w_hn = simpleUniformInitialization((hidden_size, hidden_size))

    # Initialize the biases
    if bias:
      self.b_ir = simpleUniformInitialization((hidden_size, 1)).squeeze(-1)
      self.b_iz = simpleUniformInitialization((hidden_size, 1)).squeeze(-1)
      self.b_in = simpleUniformInitialization((hidden_size, 1)).squeeze(-1)
      self.b_hr = simpleUniformInitialization((hidden_size, 1)).squeeze(-1)
      self.b_hz = simpleUniformInitialization((hidden_size, 1)).squeeze(-1)
      self.b_hn = simpleUniformInitialization((hidden_size, 1)).squeeze(-1)

  def forward(self, x: Tensor, h: Tensor) -> Tensor:
    '''
    Forward pass

    Parameters:
      x (Tensor): The input tensor
      h (Tensor): The hidden state
    '''
    # Compute the reset gate
    reset_gate = KTorch.matmul(x, self.w_ir) + (self.b_ir if self.bias else 0)
    reset_gate += KTorch.matmul(h, self.w_hr) + (self.b_hr if self.bias else 0)
    reset_gate = reset_gate.sigmoid()

    # Compute the update gate
    update_gate = KTorch.matmul(x, self.w_iz) + (self.b_iz if self.bias else 0)
    update_gate += KTorch.matmul(h, self.w_hz) + (self.b_hz if self.bias else 0)
    update_gate = update_gate.sigmoid()

    # Compute the new memory content
    new_memory_content = KTorch.matmul(x, self.w_in) + (self.b_in if self.bias else 0)
    new_memory_content += reset_gate * (KTorch.matmul(h, self.w_hn) + (self.b_hn if self.bias else 0))
    new_memory_content = new_memory_content.tanh()

    # Compute the new hidden state
    new_h = (1 - update_gate) * h + update_gate * new_memory_content
    return new_h
  
  def parameters(self):
    out = [self.w_ir, self.w_iz, self.w_in, self.w_hr, self.w_hz, self.w_hn]
    if self.bias:
      out.extend([self.b_ir, self.b_iz, self.b_in, self.b_hr, self.b_hz, self.b_hn])
    return out
