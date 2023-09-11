"""
This module provides kernels for convolution.
"""
from functools import reduce

import numpy as np
import numpy.typing as npt


class Gaussian(): 
    """
    A gaussian kernel.
    """

    def __init__(self, domain, standard_deviation, mean=None): 
        if standard_deviation == 0: 
            raise ZeroDivisionError('Standard deviation must not be zero.')
        self.domain = domain
        self.mean = mean 
        self.std = standard_deviation

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
        return 1 / np.sum(a * np.exp(-0.5 * ((t - b) / c)**2 ))

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
        is also none. 
        """
        if self.mean is not None:
            t = self.domain
            a, b, c = self.parameter
            return a * (t - b) * np.exp(-0.5*((t - b) / c)**2) / c**2
    
    @property
    def derivative_wrt_std(self): 
        """The partial derivative w.r.t standard deviation.

        The partial derivative with respect to the standard 
        deviation. Evaulated at the parameter `self.std`. 
        """
        t = self.domain 
        a, b, c = self.parameter
        return a * (t - b)**2 * np.exp(-0.5*((t - b) / c)**2) / c**3 
    
    @property
    def partial_derivatives(self): 
        """Returns list of partial derivatives.

        The list consists of both partial derivatives 
        if the gaussian kernel is given a mean. Otherwise, 
        returns a list of the partial derivative w.r.t. 
        the standard deviation. 
        """
        if self.mean is not None: 
            return [self.derivative_wrt_mean, self.derivative_wrt_std]
        return [self.derivative_wrt_std]
    

class Mixture():
    "Kernel that consists of a mixture of kernels."
    def __init__(self, weights, kernels): 
        self.weights = weights
        self.components = kernels

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
        def differentiation(w1, f, fprime, c):
            sum_fprime = np.sum(fprime)
            return w1 * (fprime - f * sum_fprime * c)
        
        result = []
        for weight, kernel in zip(self.weights, self.components):
            for derivative in kernel.partial_derivatives:
                result.append(
                    differentiation(
                        w1=weight, 
                        f=kernel.image, 
                        fprime=derivative, 
                        c=kernel.normalization_constant))
        return result

    @property 
    def partial_derivatives(self): 
        """Partial derivatives of kernel mixture. 
        
        Returns a list of the partial derivatives of the 
        kernel mixture. 
        """
        return [self.derivative_wrt_weights, self.derivative_wrt_components]
