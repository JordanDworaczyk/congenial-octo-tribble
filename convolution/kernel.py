import numpy as np 
import numpy.typing as npt

from functools import reduce

sum = np.sum 

class Gaussian(): 

    def __init__(self, domain, mean, standard_deviation): 
        if standard_deviation == 0: 
            raise ZeroDivisionError('Standard deviation must not be zero.')
        self.domain = domain
        self.mean = mean 
        self.std = standard_deviation

    @property
    def normalization_constant(self):
        return 1 / sum(gaussian(self.domain, a=1, b=self.mean, c=self.std))

    @property
    def parameter(self): 
        return self.normalization_constant, self.mean, self.std

    @property 
    def image(self):
        return gaussian(self.domain, *self.parameter)
    
    @property 
    def derivative_wrt_mean(self): 
        if self.mean == 0: 
            return np.zeros_like(self.domain)
        return gaussian_derivative_wrt_b(self.domain, *self.parameter)
    
    @property
    def derivative_wrt_std(self): 
        return gaussian_derivative_wrt_c(self.domain, *self.parameter)
    
    @property
    def partial_derivatives(self): 
        return [self.derivative_wrt_mean, self.derivative_wrt_std]
    

def gaussian(domain: npt.NDArray, a: float, b: float, c: float) -> npt.NDArray: 
    t = domain 
    return a * np.exp(-0.5 * ((t - b) / c)**2 ) 

def gaussian_derivative_wrt_a(domain: npt.NDArray, b, c):
    t = domain  
    return np.exp(-0.5 * ((t - b) / c)**2) 

def gaussian_derivative_wrt_b(domain: npt.NDArray, a, b, c):
    t = domain 
    return a * (t - b) * np.exp(-0.5*((t - b) / c)**2) / c**2 

def gaussian_derivative_wrt_c(domain: npt.NDArray, a, b, c): 
    t = domain 
    return a * (t - b)**2 * np.exp(-0.5*((t - b) / c)**2) / c**3 

class Mixture(): 
    def __init__(self, weights, kernels): 
        self.weights = weights
        self.components = kernels

    @property 
    def domain(self): 
        return reduce(np.union1d, (kernel.domain for kernel in self.components))

    @property
    def image(self): 
        weights = np.array(self.weights)
        components = np.array([kernel.image for kernel in self.components])
        return weights @ components 
    
    @property 
    def derivative_wrt_weights(self): 
        ...

    @property 
    def derivative_wrt_components(self): 
        ... 

    @property 
    def partial_derivatives(self): 
        return [self.derivative_wrt_weights, self.derivative_wrt_components]
