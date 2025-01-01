from abc import ABC, abstractmethod
from typing import Callable

class BaseUnsupervisedEstimator(ABC):
    '''
    Base class for unsupervised estimators.
    '''

    def __init__(self, base_metric: Callable):
        self._base_metric = base_metric


    @property
    def base_metric(self) -> Callable:
        return self._base_metric
    
    @abstractmethod
    def fit(self, X):
        '''
        Fit the model to the training data.
        
        Params:
          - X: Input features (numpy array or pandas DataFrame)
        '''
        pass

    @abstractmethod
    def predict(self, X):
        '''
        Generate predictions for the input data.
        
        Params:
          - X: Input features (numpy array or pandas DataFrame)
        '''
        pass

    @abstractmethod
    def transform(self, X):
        '''
        Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers.
        
        Params:
          - X: Input features (numpy array or pandas DataFrame)
        '''
        pass

    def fit_predict(self, X):
        '''
        Fit the model to the training data and generate predictions.
        
        Params:
          - X: Input features (numpy array or pandas DataFrame)
        '''
        self.fit(X)
        return self.predict(X)
    
    def fit_transform(self, X):
        '''
        Fit the model to the training data and transform it.
        
        Params:
          - X: Input features (numpy array or pandas DataFrame)
        '''
        self.fit(X)
        return self.transform(X)
    
    @abstractmethod
    def clone(self):
        '''
        Create a copy of the estimator.
        
        Returns:
          - estimator: A new instance of the estimator
        '''
        pass

    @abstractmethod
    def score(self, X):
        '''
        Calculate the base metric of the model.

        Params:
          - X: Input features (numpy array or pandas DataFrame)
        '''
        pass
    

    

