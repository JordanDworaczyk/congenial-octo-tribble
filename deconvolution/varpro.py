
from copy import deepcopy

import numpy as np 
from scipy.linalg import pinv
from scipy.optimize import fminbound

SMALL = 1e-14

def variable_projection(
    data, 
    guess, 
    jacobian, 
    regularization_parameter, 
    regularization_matrix,
    maximum_iterations
):
    def full_functional(y, x, d, L, alpha): 
        K.kernel.variables = y # CAUTION: State state of `K.kernel.variables`
        return np.linalg.norm(K.image @ x - d)**2 + alpha**2 * np.linalg.norm(L @ x)**2
        
    K = deepcopy(guess)
    L = np.identity(data.size)
    y = K.kernel.variables
    for iteration in range(maximum_iterations + 1): 
        old_kernel = K.kernel.image

        # Variable Projection 
        KL = np.block([[K.image],  [regularization_parameter * L]])
        D = np.block([data, np.zeros_like(data)])
        x = pinv(KL) @ D

        L = regularization_matrix(x) # Update Regularization Matrix
        J = jacobian(K, x)           # Update Jacobian
        
        # Gauss-Newton w/ Line Search Strategy
        residual = data - K.image @ x
        delta_y = pinv(J) @ residual
       
       # Line Search Strategy 
        beta = fminbound(
            lambda beta: full_functional(
                    y + beta * delta_y,
                    x, 
                    data, 
                    L, 
                    regularization_parameter
                ), 
            1e-12, 
            1e0
        )

        y = y + beta * delta_y # Update Nonlinear Variable
        K.kernel.variables = y # Update Convolution Operator

        # Stopping Condition 
        mean_squared_error = np.linalg.norm(old_kernel - K.kernel.image)**2 / data.size
        if  mean_squared_error < SMALL:
            break 
    return K, x, iteration