"""
This module provides kernels for convolution.
"""
from functools import reduce
import itertools

import numpy as np
import numpy.typing as npt

import logging

# logging.basicConfig(filename='example.log', level=logging.DEBUG)

class Gaussian(): 
    """
    A gaussian kernel.
    """

    def __init__(self, domain, standard_deviation, mean=None): 
        if standard_deviation == 0: 
            raise ZeroDivisionError('Standard deviation must not be zero.') 
        self.domain = domain
        self._std = standard_deviation
        self._mean = mean 

    @property
    def variables(self): 
        if self._mean is not None: 
            return self.std, self.mean 
        return [self.std]
    
    @variables.setter
    def variables(self, values): 
        if not len(self.variables) == len(values): 
            raise Exception(f'{values}')
        if self._mean is not None: 
            self.std = values[0] 
            self.mean = values[1]
        # logging.debug(f'Setting mean')
        self.std = values[0]

    @property 
    def std(self): 
        return self._std 
    
    @std.setter 
    def std(self, value): 
        if value == 0: 
            raise ZeroDivisionError('Standard deviation must not be zero.') 
        self._std = value 

    @property 
    def mean(self): 
        if self._mean is not None: 
            return self._mean
        return 0 

    @mean.setter
    def mean(self, value): 
        if  self._mean is None:
            raise Exception('This instance has no mean.')
        self._mean = value 
        
    @property
    def normalization_constant(self):
        """Returns constant which normalizes gaussian kernel.

        The kernel is considered normalized when its sum is 
        equal to one. 
        """
        t = self.domain
        a = 1 
        b = self.mean 
        c = self.std 
        return 1 / np.sum(a * np.exp(-0.5 * ((t - b) / c)**2))

    @property
    def parameter(self): 
        "Returns parameters of gaussian kernel."
        return self.normalization_constant, self.mean, self.std

    @property 
    def image(self):
        """Returns the image of the gaussian kernel. 
        
        The set of all output values that the gaussian 
        produces over its domain.
        """
        t = self.domain
        a, b, c = self.parameter
        return a * np.exp(-0.5 * ((t - b) / c)**2 ) 
    
    @property 
    def derivative_wrt_mean(self): 
        """The partial derivative w.r.t mean.

        The partial derivative of the gaussian with respect 
        to the mean. Evaulated at the parameter `self.mean`.
        If `self.mean` is none then the derivative 
        is zero.  
        """
        # TODO: FIX ME
        def differentiation(f, fprime, c):
            sum_fprime = np.sum(fprime)
            return c * fprime - f * sum_fprime * c**2
        t = self.domain
        if self._mean is not None:
            a, b, c = self.parameter
            f = np.exp(-0.5*((t - b) / c)**2)
            fprime = (t - b) * f / c**2
            return differentiation(f, fprime, a)
        return np.zeros_like(t)
    
    @property
    def derivative_wrt_std(self): 
        """The partial derivative w.r.t standard deviation.

        The partial derivative with respect to the standard 
        deviation. Evaulated at the parameter `self.std`. 
        """
        # TODO: FIX ME 
        def differentiation(f, fprime, c):
            sum_fprime = np.sum(fprime)
            return c * fprime - f * sum_fprime * c**2
        t = self.domain 
        a, b, c = self.parameter
        f = np.exp(-0.5*((t - b) / c)**2)
        fprime = (t - b)**2 * f / c**3 
        return differentiation(f, fprime, a)
    
    @property
    def partial_derivatives(self): 
        """Returns list of partial derivatives.

        The list consists of both partial derivatives 
        if the gaussian kernel is given a mean. Otherwise, 
        returns a list of the partial derivative w.r.t. 
        the standard deviation. 
        """
        if self._mean is not None: 
            return [self.derivative_wrt_std, self.derivative_wrt_mean]
        return [self.derivative_wrt_std]
    
    @property
    def __name__(self): 
        return 'Gaussian'
    

class Mixture():
    "Kernel that consists of a mixture of kernels."
    def __init__(self, weights, kernels): 
        self._weights = np.abs(weights) / np.sum(weights)
        self.components = kernels

    @property 
    def weights(self): 
        return self._weights 
    
    @weights.setter
    def weights(self, value): 
        self._weights = np.abs(value) / np.sum(np.abs(value))

    @property
    def variables(self): 
        return tuple(
            itertools.chain.from_iterable(
                [self.weights] + [kernel.variables for kernel in self.components] 
            )
        )

    @variables.setter
    def variables(self, values): 
        if not len(values) == len(self.variables): 
            raise Exception
        n = len(self.weights)
        self.weights = values[0:n]
        for kernel in self.components: 
            m = len(kernel.variables)
            kernel.variables = values[n:n + m]
            n = m + n

    @property 
    def domain(self): 
        """Domain of the mixture. 

        The union of domains for each kernel in the mixture. 
        """
        return reduce(np.union1d, (kernel.domain for kernel in self.components))

    @property
    def image(self): 
        """Image of mixture. 

        The set of all output values that the 
        kernel mixture produces over its domain.
        """
        weights = np.array(self.weights)
        components = np.array([kernel.image for kernel in self.components])
        return weights @ components 
    
    @property 
    def derivative_wrt_weights(self): 
        """The derivative w.r.t weights. 
        
        The partial derivatives of the kernel 
        mixture with respect to the weights. The 
        weights are assumed to be constant. Therfore 
        the derivative is simply the corresponding 
        components of the mixture. 
        """
        return [kernel.image for kernel in self.components]

    @property 
    def derivative_wrt_components(self): 
        """Derivative w.r.t each component in mixture.
        
        Returns a list of the partial derivatives of 
        each component with respect to each of the 
        components parameters. 
        """
        def differentiation(w, f, fprime, c):
            sum_fprime = np.sum(fprime)
            return w * (fprime - f * sum_fprime * c)
        
        result = []
        for weight, kernel in zip(self.weights, self.components):
            for derivative in kernel.partial_derivatives:
                result.append(weight * derivative)
        return result

    @property 
    def partial_derivatives(self): 
        """Partial derivatives of kernel mixture. 
        
        Returns a list of the partial derivatives of the 
        kernel mixture. 
        """
        derivatives = [self.derivative_wrt_weights, self.derivative_wrt_components]
        return list(itertools.chain.from_iterable(derivatives))
