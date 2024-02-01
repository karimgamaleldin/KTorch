from algorithms.linear_model.RidgeRegression import RidgeRegression 
from metrics.RegressionMetrics import mean_absolute_error
import numpy as np

# Create a Toy dataset
weight = 2
bias = 3
start = 0
stop = 100
step = 0.5

X = np.arange(start, stop, step, dtype=np.float64)
X = X.reshape(len(X), 1) # Convert 1D array to 2D array of shape
y = weight * X + bias + np.random.randn(len(X), 1) * 10 # Add some noise to the target values

print(len(X), len(y))
# print(X)

# Create a Linear Regression model
model = RidgeRegression()
model.fit(X, y)
print(model.algorithm_params)
# Both models produce the correct coefficients and intercept
print(model.evaluate(X, y))
print(model.evaluate(X, y, mean_absolute_error))
