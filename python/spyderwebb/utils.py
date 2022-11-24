import numpy as np
from scipy.special import erf


def gauss2dbin(x,amp,center,sigma):
    """ Make lots of Gaussian profiles."""
    # amp, center, sigma are arrays
    # x should have shape [Npix, Ngaussians]
    
    xcen = x-center.reshape(1,-1)
    dx = 1
    x1cen = xcen - 0.5*dx  # left side of bin
    x2cen = xcen + 0.5*dx  # right side of bin

    t1cen = x1cen/(np.sqrt(2.0)*sigma.reshape(1,-1))  # scale to a unitless Gaussian
    t2cen = x2cen/(np.sqrt(2.0)*sigma.reshape(1,-1))

    # For each value we need to calculate two integrals
    #  one on the left side and one on the right side

    # Evaluate each point
    #   ERF = 2/sqrt(pi) * Integral(t=0-z) exp(-t^2) dt
    #   negative for negative z
    geval_lower = erf(t1cen)
    geval_upper = erf(t2cen)

    geval = (amp*sigma).reshape(1,-1) * np.sqrt(2.0) * np.sqrt(np.pi)/2.0 * ( geval_upper - geval_lower )

    return geval
    
    
