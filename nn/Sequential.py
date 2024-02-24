from nn.Module import Module
from autograd.engine import Tensor

class Sequential(Module):
  '''
  A sequential container to stack layers
  '''
  def __init__(self, *layers: Module):
    '''
    Initialize the sequential container with the layers
    params:
      layers: list: the layers to stack
    '''
    super().__init__()
    self.layers = layers

  def forward(self, x: Tensor) -> Tensor:
    '''
    Forward pass through the layers
    '''
    for layer in self.layers:
      x = layer(x)

    return x
  
  def parameters(self):
    '''
    Return the parameters of the module
    '''
    return [param for layer in self.layers for param in layer.parameters()]