import numpy as np


def guassian_noise(percentage, signal):
    np.random.seed(1) # Control randomness 
    mu = 0 # Mean 
    std = signal.std() # Standard deviation
    N = signal.size # number of points 
    random_noise = np.random.normal(loc=mu, scale=std, size=N)
    return  random_noise * percentage