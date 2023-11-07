import numpy as np 
import math


def rectangle(domain): 
    t = domain 
    N = t.size
    return np.piecewise(
        t,
        [t < N/6, -N/6 <= t, N/6 <= t],
        [0, 1, 0])

def heavside(domain):
    return np.heaviside(domain, 1)

# TODO
def smooth(domain):
    t = np.linspace(0, 1, domain.size)
    return  ((0.25 < t) & (t < .75))*np.sin(2*math.pi*(t - .25))**4

def mix(domain): 
    t = np.linspace(0, 1, domain.size)
    return 0.75*((0.1 < t) & (t < 0.25)) + 0.25*((0.3 < t) & (t < 0.32)) + ((0.5 < t) & (t < 1))*np.sin(2*math.pi*t)**4
