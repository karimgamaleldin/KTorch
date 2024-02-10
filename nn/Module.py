from autograd.engine import Tensor

class Module:
  '''
  A class that represents a module in a neural network
  '''
  def __call__(self, x: Tensor) -> Tensor:
    '''
    Perform the forward pass of the module
    '''
    return self.forward(x)

  def forward(self, x: Tensor) -> Tensor:
    '''
    Perform the forward pass of the module
    '''
    raise NotImplementedError

  def parameters(self):
    '''
    Return the parameters of the module
    '''
    return []