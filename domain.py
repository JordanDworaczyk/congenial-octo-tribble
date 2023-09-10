import numpy as np 


def equispaced_points(number_of_points):
    """Returns an equally spaced array centered about zero
    with `number_of_points + 1` points."""
    return np.arange(
        -number_of_points // 2,
        number_of_points // 2 + 1,
        dtype=float)