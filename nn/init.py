import torch
import math

def calculate_gain(nonlinearity: str, param=None):
  '''
  Return the recommended gain value for a function based on the pytorch documentation
  ----------------------------------------------------------------
  nonlinearity           gain 
  Linear/Identity        1

  '''
  # Checking if the nonlinearity is supported
  if nonlinearity == 'linear':
    return 1
  # to be continued while building ktorch
  return -1

def _calculate_fan_in_and_fan_out(tensor: torch.Tensor):
  '''
  Calculate the fan_in and fan_out of tensors to be used for initialization functions
  '''
  # Checking if the tensor is 1D or 0D
  if tensor.dim() < 2:
    raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

  # using pytorch covention that the input_features is the second dimension
  fan_in = tensor.shape[1] 
  fan_out = tensor.shape[0]

  # todo: check if the tensor is 3D or 4D and calculate the fan_in and fan_out accordingly
  return fan_in, fan_out

def uniform(tensor: torch.Tensor, a:float=0, b:float=1):
  '''
  Fill the input tensor with values sampled from a uniform distribution with the range [a, b]
  '''
  with torch.inference_mode():
    return tensor.uniform_(a, b)


# incomplete
def kaiming_uniform(tensor: torch.Tensor, a:float=0, mode:str='fan_in', nonlinearity:str='leaky_relu'):
  '''
  Fill the input tensor with values using the Kaiming Uniform initialization method as described in:
    Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015)

  '''
  fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
  fan = fan_in if mode == 'fan_in' else fan_out
  gain = calculate_gain(nonlinearity, a)
  bound = gain * math.sqrt(3.0 / fan)

  with torch.inference_mode():
    return tensor.uniform_(-bound, bound)
  

