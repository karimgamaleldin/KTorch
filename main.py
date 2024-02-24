from algorithms.neighbors.KNeighborsRegressor import KNeighborsRegressor
import numpy as np
from sklearn.neighbors import KNeighborsRegressor as KNeighborsRegressor_sklearn

# Create a KNeighborsRegressor model
knn_regressor = KNeighborsRegressor(n_neighbors=2, weights='distance', p=3, metric='minkowski')
knn2 = KNeighborsRegressor_sklearn(n_neighbors=2, weights='distance', p=3, metric='manhattan')

# Create a random dataset
np.random.seed(0)
X = np.array([[3, 4, 5], [6, 7, 8], [0, 1, 2]])
y = np.array([100, 200, 300])
print(X.shape)

# Fit the model to the dataset
knn_regressor.fit(X, y)
knn2.fit(X, y)

# Predict the target values for the input features
X_test = np.array([[1, 2, 3], [4, 5, 6]])
y_pred = knn_regressor.predict(X_test)
y_pred2 = knn2.predict(X_test)
print(y_pred)
print(y_pred2)
