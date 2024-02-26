from algorithms.linear_model.LinearRegression import LinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
import numpy as np

# Create the data
X = np.array([[0, 3, 5], [1, 2, 6], [8, 5, 4], [2, 6, 48], [43, 45, 75]])
print(X)
y = 4 * np.sum(X, axis=1, keepdims=True) + 3 # + np.random.randn(5).reshape(-1, 1)
print(y)

# Create an instance of the LinearRegression class
my = LinearRegression()
sklearn = SklearnLinearRegression()

# Fit the models
my.fit(X, y)
sklearn.fit(X, y)

# Predict the target values
my_predictions = my.predict(X)
sklearn_predictions = sklearn.predict(X)

# Compare the predictions
print('My predictions:', my_predictions)

print('Sklearn predictions:', sklearn_predictions)

# Compare the weights and bias
print('My weights:', my.weights, 'My bias:', my.bias)
print('Sklearn weights:', sklearn.coef_, 'Sklearn bias:', sklearn.intercept_)
