import numpy as np 

from scipy.linalg import toeplitz as _toeplitz


def toeplitz(c, k): 
    n = c.size
    col = np.zeros_like(c)
    row = np.zeros_like(c)
    col[0:k+1] = c[k::1]
    row[0:k+1] = c[k::-1]
    return _toeplitz(col, row)

