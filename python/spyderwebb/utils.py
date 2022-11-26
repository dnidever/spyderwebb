import numpy as np
from scipy.special import erf
from functools import wraps
from scipy import interpolate

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

def weightedregression(x,y,w,axis=0,zero=False):
    """ Perform weighted simple linear regression on 2D data."""
    # zero means the intercept = 0, y = m*x
    
    if x.ndim==1:
        axis = None
    n = np.nansum(np.isfinite(x)*np.isfinite(y),axis=axis)

    # Relative weights
    #  only count x/y points that are not NaN
    totwt = np.nansum(w*np.isfinite(x)*np.isfinite(y),axis=axis)
    wt = w/totwt
            
    # Solve for intercept as well
    if zero==False:
        # Simple Linear Regression
        # https://en.wikipedia.org/wiki/Simple_linear_regression
        # y = m*x+b
        # m = (N*Sum(x*y)-Sum(x)*Sum(y))/(N*Sum(x^2)-Sum(x)^2)
        
        # Weighted version
        # https://stats.stackexchange.com/questions/12673/formula-for-weighted-simple-linear-regression
        # xwtmn = Sum(w*x)/Sum(w)
        # ywtmn = Sum(w*y)/Sum(w)
        # m = Sum(w*(x-xwtmn)*(y-ywtmn)) / Sum(w*(x-xwtmn)^2)
        # b = ywtmn - m*xwtmn

        # https://ms.mcmaster.ca/canty/teaching/stat3a03/Lectures7.pdf
        
        xwtmn = np.nansum(wt*x,axis=axis)
        ywtmn = np.nansum(wt*y,axis=axis)
        m = np.nansum(wt*(x-xwtmn)*(y-ywtmn),axis=axis) / np.nansum(wt*(x-xwtmn)**2,axis=axis)
        b = ywtmn - m*xwtmn

        # Uncertainties
        # var(m) = sig^2 / Sum(w*(x-xwtmn)^2)
        # var(b) = sig^2 * (1/Sum(w) + xwtmn^2/Sum(w*(x-xwtmn)^2))
        sig = np.sqrt( np.nansum(wt*(y-b-m*x)**2,axis=axis)/(n-2) )
        merr = np.sqrt( sig**2 / np.nansum(wt*(x-xwtmn)**2,axis=axis) )
        berr = np.sqrt( sig**2 * (1 + xwtmn**2/np.nansum(wt*(x-xwtmn)**2,axis=axis)) )
        
        return m,merr,b,berr

    # Intercept is zero
    else:
        # simple linear regression without the regression term (single regressor)
        # https://en.wikipedia.org/wiki/Simple_linear_regression
        m = np.nansum(wt*x*y,axis=axis)/np.nansum(wt*x**2,axis=axis)
        sig = np.sqrt( np.nansum(wt*(y-m*x)**2,axis=axis)/(n-1) )
        merr = np.sqrt( sig**2 / np.nansum(wt*x**2,axis=axis) )
        
        return m,merr

def scalarDecorator(func):
    """Decorator to return scalar outputs for wave2pix and pix2wave
    """
    @wraps(func)
    def scalar_wrapper(*args,**kwargs):
        if np.array(args[0]).shape == ():
            scalarOut= True
            newargs= (np.array([args[0]]),)
            for ii in range(1,len(args)):
                newargs= newargs+(args[ii],)
            args= newargs
        else:
            scalarOut= False
        result= func(*args,**kwargs)
        if scalarOut:
            return result[0]
        else:
            return result
    return scalar_wrapper

@scalarDecorator
def wave2pix(wave,wave0):
    """ convert wavelength to pixel given wavelength array
    Args :
       wave(s) : wavelength(s) (\AA) to get pixel of
       wave0 : array with wavelength as a function of pixel number 
    Returns :
       pixel(s) in the chip
    """
    pix0 = np.arange(len(wave0))
    # Need to sort into ascending order
    sindx = np.argsort(wave0)
    wave0 = wave0[sindx]
    pix0 = pix0[sindx]
    # Start from a linear baseline
    #baseline = np.polynomial.Polynomial.fit(wave0,pix0,1)
    #ip = interpolate.InterpolatedUnivariateSpline(wave0,pix0/baseline(wave0),k=3)
    #out = baseline(wave)*ip(wave)
    out = interpolate.InterpolatedUnivariateSpline(wave0,pix0,k=3)(wave) 
    # NaN for out of bounds
    out[wave > wave0[-1]] = np.nan
    out[wave < wave0[0]] = np.nan
    return out

@scalarDecorator
def pix2wave(pix,wave0):
    """ convert pixel(s) to wavelength(s)
    Args :
       pix : pixel(s) to get wavelength at
       wave0 : array with wavelength as a function of pixel number 
    Returns :
       wavelength(s) in \AA
    """
    pix0 = np.arange(len(wave0))
    # Need to sort into ascending order
    sindx = np.argsort(pix0)
    wave0 = wave0[sindx]
    pix0 = pix0[sindx]
    # Start from a linear baseline
    baseline = np.polynomial.Polynomial.fit(pix0,wave0,1)
    ip = interpolate.InterpolatedUnivariateSpline(pix0,wave0/baseline(pix0), k=3)
    out = baseline(pix)*ip(pix)
    # NaN for out of bounds
    out[pix < 0] = np.nan
    out[pix > 2047] = np.nan
    return out
