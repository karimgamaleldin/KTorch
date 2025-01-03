import numpy as np 

def cosine_similarity(X: np.ndarray, Y:np.ndarray) -> float:
    '''
    Calculate the cosine similarity between two vectors
    
    Params:
      - X: First vector (numpy array)
      - Y: Second vector (numpy array)
      
      Returns:
        - cosine_similarity: Cosine similarity between the two vectors (float)
    '''
    dot_product = np.dot(X, Y)
    norm_X = np.linalg.norm(X)
    norm_Y = np.linalg.norm(Y)
    return dot_product / (norm_X * norm_Y)