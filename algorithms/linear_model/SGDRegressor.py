from core.BaseEstimator import BaseEstimator
from utils.RegressionMetrics import mean_squared_error, huber, epsilon_insensitive, squared_epsilon_insensitive
import numpy as np

class SGDRegressor(BaseEstimator):
    '''
    Stochastic Gradient Descent Regressor
    
    A linear regression model that uses stochastic gradient descent to optimize the model parameters.
    
    It supports Ridge (L2), Lasso (L1) regularization and Elastic Net (L1 + L2) regularization.
    '''

    def __init__(self, loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, max_iter=1000, fit_intercept=True, lr=0.01, power_t=0.25, epsilon=0.1, tol=1e-5):
        '''
        Initialize the SGDRegressor model with the loss function, penalty, regularization strength, and other parameters.
        '''
        super().__init__('SGD Regressor', 'linear_model', mean_squared_error)
        self.loss = loss
        assert self.loss in ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], "loss must be one of 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'"
        self.penalty = penalty
        assert self.penalty in ['l1', 'l2', 'elasticnet', 'none'], "penalty must be one of 'l1', 'l2', 'elasticnet', 'none'"
        self.alpha = alpha
        assert self.alpha >= 0, "alpha must be greater than or equal to 0"
        self.l1_ratio = l1_ratio
        assert 0 <= self.l1_ratio <= 1, "l1_ratio must be between 0 and 1"
        self.max_iter = max_iter
        assert self.max_iter > 0, "max_iter must be greater than 0"
        self.fit_intercept = fit_intercept
        self.lr = lr
        assert self.lr > 0, "eta0 must be greater than 0"
        self.power_t = power_t
        assert self.power_t >= 0, "power_t must be greater than or equal to 0"
        self.epsilon = epsilon
        assert self.epsilon >= 0, "epsilon must be greater than or equal to 0"
        self.coef_ = None
        self.intercept_ = None
        self.tol = tol

    def fit(self, X, y):
        # Ensure X and y are numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Initialize the weights
        n_samples, n_features = X.shape
        self.coef_ = np.random.randn(n_features, 1)
        self.intercept_ = 0.0

        # Fit the model
        prev_loss = np.inf
        for i in range(self.max_iter):
            for j in range(n_samples):
                X_j = X[j].reshape(-1, 1)
                y_j = y[j]
                y_pred = X_j.T @ self.coef_ + self.intercept_
                loss = self.loss_function(y_j, y_pred)
                gradient = self.gradient(y_j, y_pred)
                self.coef_ -= self.lr * (gradient * X_j + self.regularization_gradient())
                self.intercept_ -= self.lr * gradient
                if np.abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss
        print('SGD Regressor model fitted successfully')

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

    def evaluate(self, X, y, metric=None):
        if metric is None:
            metric = self.base_metric
        predictions = self.predict(X)
        return metric(y, predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)

    def clone(self):
        return SGDRegressor(self.loss, self.penalty, self.alpha, self.l1_ratio, self.max_iter, self.fit_intercept, self.learning_rate, self.lr, self.power_t, self.epsilon)

    def loss_function(self, y_true, y_pred):
        '''
        Compute the loss function for the given true and predicted values.
        '''
        loss = self.regularization_term()
        if self.loss == 'squared_loss':
            loss += mean_squared_error(y_true, y_pred)
        elif self.loss == 'huber':
            loss += huber(y_true, y_pred, self.epsilon)
        elif self.loss == 'epsilon_insensitive':
            loss += epsilon_insensitive(y_true, y_pred, self.epsilon)
        elif self.loss == 'squared_epsilon_insensitive':
            loss += squared_epsilon_insensitive(y_true, y_pred, self.epsilon)
        return loss

    def gradient(self, y_true, y_pred):
        '''
        Compute the gradient of the loss function with respect to the prediction
        '''
        grad = 0
        if self.loss == 'squared_loss':
            grad += 2 * (y_pred - y_true)
        elif self.loss == 'huber':
            error = y_true - y_pred
            grad += self.epsilon * np.sign(error) if np.abs(error) <= self.epsilon else error
        elif self.loss == 'epsilon_insensitive':
            error = y_true - y_pred
            grad += self.epsilon * np.sign(error) if np.abs(error) <= self.epsilon else 0
        elif self.loss == 'squared_epsilon_insensitive':
            error = y_true - y_pred
            grad += 2 * self.epsilon * error if np.abs(error) <= self.epsilon else 0
        return grad

    def regularization_gradient(self):
        '''
        Compute the gradient of the regularization term with respect to the weights
        '''
        if self.penalty == 'l1':
            return self.alpha * np.sign(self.coef_)
        elif self.penalty == 'l2':
            return self.alpha * self.coef_
        elif self.penalty == 'elasticnet':
            return self.l1_ratio * np.sign(self.coef_) + (1 - self.l1_ratio) * self.coef_
        return 0

    def regularization_term(self):
        '''
        Compute the regularization term for the weights
        '''
        if self.penalty == 'l1':
            return self.alpha * np.sum(np.abs(self.coef_))
        elif self.penalty == 'l2':
            return 0.5 * self.alpha * np.sum(self.coef_ ** 2)
        elif self.penalty == 'elasticnet':
            return self.l1_ratio * np.sum(np.abs(self.coef_)) + 0.5 * (1 - self.l1_ratio) * np.sum(self.coef_ ** 2)
        return 0

