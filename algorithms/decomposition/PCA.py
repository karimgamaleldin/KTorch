from core.BaseEstimator import BaseEstimator
import numpy as np


class PCA(BaseEstimator):
    """
    Principal Component Analysis

    An unsupervised learning algorithm that is used to reduce the dimensionality of the data. It works by finding the directions of maximum variance in high-dimensional data and projecting it onto a new subspace with fewer dimensions.
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Zeroing the mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        cov_matrix = np.cov(X.T)
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eigenvectors = eigenvectors.T

        # Sorting the eigenvectors by decreasing eigenvalues
        idx = np.argsort(eigenvalues)[::-1]

        # Getting the top n_components eigenvectors
        eigenvectors = eigenvectors[idx]
        eigenvalues = eigenvalues[idx]
        self.components = eigenvectors[: self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def clone(self):
        return PCA(n_components=self.n_components)
