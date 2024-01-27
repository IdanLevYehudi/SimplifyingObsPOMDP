# Based on following blog post:
# http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
# The following could be replaces by scipy's qmc

import numpy as np
from functools import lru_cache

# Using the above nested radical formula for g=phi_d
# or you could just hard-code it.
# phi(1) = 1.6180339887498948482
# phi(2) = 1.32471795724474602596


@lru_cache
def phi(d, num_iter=30):
    x = 2.0000
    for i in range(num_iter):
        x = pow(1 + x, 1/(d + 1))
    return x


def rd_sequence(dim, start_n, end_n=None, scale=None, shift=None, seed=0.5):
    # Seed can be any real number.
    # Common default setting is typically seed=0
    # But seed = 0.5 is generally better.
    
    # This method can be used to incrementally add more points, by changing accordingly 
    # start_n and end_n. The range of points is taken between [start_n+1, end_n+1).
    # If end_n is not provided, it is assumed the range of points is [1, start_n+1).

    if end_n is None:
        end_n = start_n
        start_n = 0
    
    g_inv = 1 / phi(dim)
    exponents = np.arange(start=1, stop=dim + 1)
    alpha = np.power(g_inv, exponents)
    points = np.zeros(shape=(end_n - start_n, dim))
    points = np.multiply(
        np.arange(start=start_n + 1, stop=end_n + 1).reshape(-1, 1), alpha)
    points = np.mod(points, 1)

    if scale is not None:
        points = np.multiply(points, scale)
    if shift is not None:
        points += shift
    return points
