from algorithms.linear_model.LinearRegression import LinearRegression
from sklearn.linear_model import LinearRegression as LinearRegression_sklearn
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
y = weight * X + bias

print(len(X), len(y))
# print(X)

# Create a Linear Regression model
model = LinearRegression()
model_sklearn = LinearRegression_sklearn()
model.fit(X, y)
model_sklearn.fit(X, y)
print(model_sklearn.coef_, model_sklearn.intercept_)
print(model.algorithm_params)

# Both models produce the correct coefficients and intercept
print(model.evaluate(X, y))
print(model.evaluate(X, y, mean_absolute_error))
