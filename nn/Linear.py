import torch
import math
from nn.init import _calculate_fan_in_and_fan_out, uniform

class Linear:

  def __init__(self, in_features, out_features, bias=True):
    self.in_features = in_features
    self.out_features = out_features
    self.shape = (out_features, in_features)
    fan_in, _ = _calculate_fan_in_and_fan_out(self.shape) # get the fan in and the fan out for the weights initialization
    bound = 1 / math.sqrt(fan_in) # calculate the bound for the weights initialization
    self.weight = uniform(torch.empty(self.shape), -bound, bound) # initialize the weights using the uniform distribution
    self.bias = uniform(torch.empty(out_features), -bound, bound) if bias else None # initialize the bias using the uniform distribution if bias is True

def __call__(self, x):
  self.out = torch.matmul(x, self.weight.t())
  if self.bias is not None:
    self.out += self.bias 
  return self.out 

def parameters(self):
  return [self.weight] + ([self.bias] if self.bias is not None else []) 
