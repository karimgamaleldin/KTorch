from algorithms.neighbors.KNeighborsClassifier import KNeighborsClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN

# Create a KNeighborsRegressor model
knn_regressor = KNeighborsClassifier(n_neighbors=3, weights='distance', p=3, metric='minkowski')
knn2 = KNN(n_neighbors=3, weights='distance', p=3, metric='manhattan')

# Create a random dataset
np.random.seed(0)
X = np.array([[3, 4, 5], [6, 7, 8], [0, 1, 2]])
y = np.array([1, 1, 2])
print(X.shape)

# Fit the model to the dataset
knn_regressor.fit(X, y)
knn2.fit(X, y)

# Predict the target values for the input features
X_test = np.array([[1, 2, 3], [4, 5, 6]])
y_test = np.array([1, 2])
y_pred = knn_regressor.predict(X_test)
y_pred2 = knn2.predict(X_test)
print(y_pred)
print(y_pred2)

# print(knn_regressor.score(X_test, y_test))
# print(knn2.score(X_test, y_test))
