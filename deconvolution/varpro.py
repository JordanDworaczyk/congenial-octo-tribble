
from copy import deepcopy

import numpy as np 
from scipy.linalg import pinv
from scipy.sparse import spdiags
from scipy.optimize import fminbound

SMALL = 1e-16


def total_variation(x): 
    BETA = 1e-16
    D = forward_difference_matrix(x)
    return np.diag((1/(((D @ x)**2 + BETA**2)**(1/4)))) @ D


def forward_difference_matrix(data): 
    return spdiags([-np.ones_like(data), np.ones_like(data)], [0, 1], data.size, data.size).toarray()

def variable_projection(
    data, 
    operator_guess, 
    jacobian, 
    regularization_parameter, 
    regularization_matrix,
    maximum_iterations,
    signal_guess=None,
    change=1e-16

):
    def full_functional(y, x, d, L, alpha): 
        K.kernel.variables = y # CAUTION: State state of `K.kernel.variables`
        return np.linalg.norm(K.image @ x - d)**2 + alpha**2 * np.linalg.norm(L @ x)**2
        
    K = deepcopy(operator_guess)
    if signal_guess is not None: 
        L = regularization_matrix(signal_guess)
    else: 
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
            SMALL, 
            1
        )

        y = y + beta * delta_y # Update Nonlinear Variable
        K.kernel.variables = y # Update Convolution Operator

        # Stopping Condition 
        mean_squared_error = np.linalg.norm(old_kernel - K.kernel.image)**2 / data.size
        if  mean_squared_error < change:
            break 
    return K, x, iteration, mean_squared_error
