from functools import partial
from copy import copy 
from typing import Callable

import numpy as np 
import numpy.typing as npt
import matplotlib.pyplot as plt

from scipy.linalg import toeplitz, pinv
from scipy.sparse import spdiags
from scipy.optimize import fminbound


def equispaced_points(number_of_points):
    """Returns an equally spaced array centered about zero
    with `number_of_points + 1` points."""
    return np.arange(
        -number_of_points // 2,
        number_of_points // 2 + 1,
        dtype=float)

def toeplitz_structure(c, k): 
    n = c.size
    col = np.zeros_like(c)
    row = np.zeros_like(c)
    col[0:k+1] = c[k::1]
    row[0:k+1] = c[k::-1]
    return toeplitz(col, row)

class BlurringOperator(): 
    def __init__(self, kernel, structure):
        self.kernel = kernel
        self.structure = structure

    @property    
    def matrix(self):
        return self.structure(self.kernel, self.kernel.size // 2)
    
    def __matmul__(self, other): 
        return self.matrix @ other

class Kernel(): 
    def __init__(self, 
                 domain: npt.NDArray, 
                 parameters: list, 
                 function: Callable, 
                 derivatives: list[Callable]): 
        self.domain: npt.NDArray = domain 
        self.parameter: list = parameters 
        self.function: Callable = function 
        self.derivative: dict[Callable] = derivatives

class Mixture(): 
    def __init__(
            self, 
            domain, 
            mixture_weights, 
            mixture_component_parameters, 
            mixture_components, 
            mixture_component_derivatives
        ):
        self.domain = domain 
        self.weights = mixture_weights 
        self.parameters = mixture_component_parameters 
        self.components = [partial(component, domain) for component in mixture_components]
        self.component_derivatives = [partial(derivative, domain) for derivative in mixture_component_derivatives]
    
    @property
    def array(self): 
        return self.weights @ self.components_array        
    
    @property
    def components_array(self): 
        return [component(parameter) for parameter, component in zip(self.parameters, self.components)]

    @property
    def component_derivatives_array(self): 
        result = []
        for derivative, parameter in zip(self.component_derivatives, self.parameters): 
            result.append(derivative(*parameter))
        return np.array(result)

    @property
    def partial_derivative_wrt_parameters(self):
        result = []
        for weight in self.weights:
            for component, component_derivative, parameter in zip(self.components, self.component_derivatives, self.parameters):
                component = component(parameter)
                component_derivative = component_derivative(parameter)
                normalization_constant = sum(component)
                sum_derivative = sum(component_derivative)
                result.append(
                    weight * (component_derivative - (component * sum_derivative / normalization_constant ))
                )
        return result 

    @property
    def flatten(self):
        return np.hstack((self.weights, self.parameters.flatten()))

    @property
    def partial_derivative_wrt_weights(self): 
        return [component(parameter) for parameter, component in zip(self.parameters, self.components)]

    @property 
    def partial_derivatives(self):
        return self.partial_derivative_wrt_weights + self.partial_derivative_wrt_parameters

    def visualize_mixture(self, title): 
        fig, ax = plt.subplots()
        ax.plot(self.domain, self.array, 'k', label='Mixture Distribution')
        for weight, component_array, component in zip(self.weights, self.components_array, self.components):
            ax.plot(self.domain, weight * component_array, 'k:', label=component.func.__name__)
        ax.set_xlabel('Pixels')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        plt.title(title)

    def update(self, weights, parameters):
        self.weights = weights 
        self.parameters = parameters.reshape(self.parameters.shape) 

def total_variation(x): 
    BETA = 1e-16
    differential_operator = forward_difference_matrix(x)
    return np.diag((1/(((differential_operator*x)**2 + BETA**2)**(1/4)))) @ differential_operator

def forward_difference_matrix(data): 
    return spdiags([-np.ones_like(data), np.ones_like(data)], [0, 1], data.size, data.size)

def rjf_jacobian(signal, partial_derivatives, structure): 
    result = []
    for partial_derivative in partial_derivatives:
            block = structure(partial_derivative, partial_derivative.size // 2)
            result.append(
                 block @ signal 
            )
    return np.array(result).T

def objective_function(blurring_operator, signal, data, regularization_parameter, regularization_matrix):
    norm = np.linalg.norm 
    K = blurring_operator
    x = signal 
    d = data 
    gamma = regularization_parameter
    L = regularization_matrix
    return norm(K @ x - d) ** 2 + gamma ** 2 * norm(L @ x) ** 2

def line_search(alpha, function, delta_y, mixture_model, blurring_operator): 
    y = np.hstack((mixture_model.weights, mixture_model.parameters.flatten()))
    line = y + alpha * delta_y
    mixture_model.weights = line[0:mixture_model.weights.size] 
    parameters = line[mixture_model.weights.size:mixture_model.weights.size + 1 + mixture_model.parameters.size]
    mixture_model.parameters = parameters.reshape(mixture_model.parameters.shape)
    blurring_operator.kernel = mixture_model.array
    return function(blurring_operator)


def variable_projection(
        data, 
        initial_signal,
        initial_mixture_model,
        blur_structure,
        jacobian,
        regularization_parameter,
        regularization_matrix,
        max_iterations
    ): 
    L = regularization_matrix(initial_signal)
    mixture_model = copy(initial_mixture_model)
    kernel_partial_derivatives = mixture_model.partial_derivatives


    y = mixture_model.flatten
    for iteration in range(max_iterations): 
        # update blurring operator
        K = BlurringOperator(
            kernel = mixture_model.array, 
            structure=blur_structure)

        # reduce full functional
        KL = np.block([[K.matrix], [regularization_parameter * L]])
        bigd = np.block([data, np.zeros_like(data)])
        signal = pinv(KL) @ bigd

        # update jacobian
        J = jacobian(signal, kernel_partial_derivatives, blur_structure)
        
        # Gauss-Newton method for least squares 
        residual = data - K @ signal
        delta_y = pinv(J) @ residual
        
        # line search strategy
        full_functional = partial(objective_function, signal=signal, data=data, regularization_parameter=regularization_parameter, regularization_matrix=L)
        search = partial(line_search, function=full_functional, delta_y=delta_y, mixture_model=mixture_model, blurring_operator=K)
 
        alpha = fminbound(search, 0, 1)
        y = y + alpha * delta_y

        # reshape
        weights = y[0:mixture_model.weights.size]
        parameters = y[mixture_model.weights.size:mixture_model.weights.size + 1 + mixture_model.parameters.size]
        
        # normalize weights 
        sum_weights = np.sum(np.abs(weights))
        weights = np.abs(weights) / sum_weights

        mixture_model.update(weights, parameters)

        if np.linalg.norm(residual, 1) / residual.size < 1e-2:
            break

    return mixture_model, signal, iteration + 1
