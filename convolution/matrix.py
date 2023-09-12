import numpy as np 

from scipy.linalg import toeplitz as _toeplitz


def toeplitz(c, k): 
    n = c.size
    col = np.zeros_like(c)
    row = np.zeros_like(c)
    col[0:k+1] = c[k::1]
    row[0:k+1] = c[k::-1]
    return _toeplitz(col, row)

class Operator(): 

    def __init__(self, kernel, structure): 
        self.kernel = kernel 
        self.structure = structure 

    @property
    def image(self):
        return self.structure(self.kernel.image, self.kernel.image.size // 2)

    @property
    def partial_derivatives(self): 
        result = list()
        for derivatives in self.kernel.partial_derivatives: 
            for derivative in derivatives: 
                result.append(
                    self.structure(
                        derivative, 
                        derivative.size // 2
                    )
                )
        return result

