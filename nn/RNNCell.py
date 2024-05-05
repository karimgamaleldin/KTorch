from autograd.engine import Tensor
from core import KTorch
from nn.Module import Module
from nn.Tanh import Tanh
from nn.ReLU import ReLU
from nn.init import simpleUniformInitialization

class RNNCell(Module):
  '''
  A class that represents a single RNN cell
  '''
  def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = 'tanh'):
    '''
    Initialize the RNN cell
    '''
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias
    self.nonlinearity = nonlinearity
    self.w_ih = simpleUniformInitialization((hidden_size, input_size)) # to initialize correctly we need the order to be (output_size, input_size)
    self.w_ih = self.w_ih.transpose(0, 1) # transpose the weight matrix, just instead of transposing it in the forward pass
    self.w_hh = simpleUniformInitialization((hidden_size, hidden_size))
    self.b_ih = simpleUniformInitialization((1, hidden_size)) if bias else None
    self.b_hh = simpleUniformInitialization((1, hidden_size)) if bias else None
    self.b_ih = self.b_ih.transpose(0, 1) if bias else None # transpose the bias matrix, just instead of transposing it in the forward pass
    self.b_hh = self.b_hh.transpose(0, 1) if bias else None # transpose the bias matrix, just instead of transposing it in the forward pass
    assert nonlinearity in ['tanh', 'relu'], 'Nonlinearity must be either tanh or relu'
    self.activation_function = Tanh() if nonlinearity == 'tanh' else ReLU()
    self.hidden = None
    
  def forward(self, x: Tensor, hidden: Tensor) -> Tensor:
    '''
    Perform the forward pass of the RNN cell
    '''
    term_1 = KTorch.matmul(x, self.w_ih) + self.b_ih if self.bias else KTorch.matmul(x, self.w_ih)
    term_2 = KTorch.matmul(hidden, self.w_hh) + self.b_hh if self.bias else KTorch.matmul(hidden, self.w_hh)
    out = self.activation_function(term_1 + term_2)
    self.hidden = out
    return out

  def parameters(self):
    '''
    Return the parameters of the RNN cell
    '''
    if self.bias:
      return [self.w_ih, self.w_hh, self.b_ih, self.b_hh]
    return [self.w_ih, self.w_hh]
  
