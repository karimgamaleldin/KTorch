import numpy as np
import pandas as pd 
from typing import Union

def validate_convert_to_numpy(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Validate the input features and convert them to a numpy array if they are in a pandas DataFrame format.
    
    Params:
      - X: Input features (numpy array or pandas DataFrame)
    
    Returns:
      - X: Input features in numpy array format
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if not isinstance(X, np.ndarray):
        raise ValueError("Input features must be a numpy array or pandas DataFrame")
    return X