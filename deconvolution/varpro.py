



def variable_projection(
    data, 
    guess, 
    regularization_parameter, 
    regularizatoin_matrix,
    maximum_iterations
):
    def full_functional(y, x, d, L, alpha): 
        K.kernel.variables = y 
        return np.linalg.norm(K.image @ x - d) + alpha**2 * np.linalg.norm(L @ x)
    K = deepcopy(guess)
    L = regularizatoin_matrix()
    y = K.kernel.variables
    for iteration in range(maximum_iterations): 
        KL = np.block([[K.image],  [regularization_parameter * L]])
        D = np.block([data, np.zeros_like(data)])

        x = pinv(KL) @ D
        J = rjf_jacobian(K, x)
        residual = data - K.image @ x
        delta_y = pinv(J) @ residual



        beta = fminbound(lambda beta: full_functional(y + beta * delta_y, x, data, L, regularization_parameter), 1e-3, 1e3)

        y = y + beta * delta_y 
        old_kernel = deepcopy(K.kernel)
        K.kernel.variables = y 
        # sum_weights = np.sum(K.kernel.weights)
        # K.kernel.weights = np.abs(K.kernel.weights) / sum_weights
        if np.linalg.norm(old_kernel.image - K.kernel.image, np.inf) < 1e-14:
            print(np.linalg.norm(old_kernel.image - K.kernel.image, np.inf) )
            print(y)
            break 
                

        
    return K, x, iteration + 1