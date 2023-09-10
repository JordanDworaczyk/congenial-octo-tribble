import numpy as np 


def rectangle(domain): 
    t = domain 
    N = t.size
    return np.piecewise(
        t,
        [t < N/6, -N/6 <= t, N/6 <= t],
        [0, 1, 0])