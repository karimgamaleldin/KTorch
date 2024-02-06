import numpy as np
from nn.Linear import Linear
# Assuming nn.init.simpleUniformInitialization and auto_grad.engine.Parameter are correctly imported

# Create a linear layer with 3 input features and 2 output features
layer = Linear(3, 1)

# Print the weights and bias of the layer
print(layer.parameters())

# Create a random input tensor with 3 features
x = np.array([[1, 2, 3]])

# Perform the forward pass of the linear layer

out = layer(x)

# Print the output of the linear layer
print('out', out)

# Perform the backward pass of the linear layer
grad = np.array([[1]])
dx = layer.backward(grad)

# Print the gradient of the input tensor
print('dx', dx)

# Print the gradients of the parameters
print(layer.parameters())

# Expected output:
# [array([[-0.068,  0.019,  0.022],
#        [ 0.068, -0.019, -0.022]]), array([0.068, 0.019])]
# out [ 0.014 -0.014]
# dx [ 0.068 -0.019 -0.022]
# [array([[0.068, 0.019, 0.022],
#        [0.068, 0.019, 0.022]]), array([1, 1])]

