import math
import numpy as np

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

def _calculate_fan_in_and_fan_out(shape):
  '''
  Calculate the fan_in and fan_out of tensors to be used for initialization functions
  '''
  # Checking if the tensor is 1D or 0D
  if len(shape) < 2:
    raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

  # using pytorch covention that the input_features is the second dimension
  fan_in = shape[1] 
  fan_out = shape[0]

  # todo: check if the tensor is 3D or 4D and calculate the fan_in and fan_out accordingly
  return fan_in, fan_out

def simpleUniformInitialization(shape, fan='fan_in', out_shape=None):
  '''
  Fill the input tensor with values sampled from a uniform distribution with the range [a, b]
  '''
  fan_in, fan_out = _calculate_fan_in_and_fan_out(shape) # get the input and output features
  fan = fan_in if fan == 'fan_in' else fan_out # get the fan value that we want to use
  bound = 1 / math.sqrt(fan) # calculate the bound
  if out_shape is not None:
    return np.random.uniform(-bound, bound, out_shape)
  else:
    return np.random.uniform(-bound, bound, shape)
  


