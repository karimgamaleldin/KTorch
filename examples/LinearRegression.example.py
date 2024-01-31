from algorithms.linear_model.LinearRegression import LinearRegression
import numpy as np

# Create a Toy dataset
weight = 2
bias = 3
start = 0
stop = 100
step = 0.5

X = np.arange(start, stop, step)
y = weight * X + bias

print(X)
print(y)