import os
import numpy as np
from scipy.special import erf
from functools import wraps
from scipy import interpolate
from scipy.ndimage import median_filter,generic_filter

def datadir():
    """ Return the data/ directory."""
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = codedir+'/data/'
    return datadir

def nanmedfilt(x,size,mode='reflect'):
    return generic_filter(x, np.nanmedian, size=size)

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


def expand_msa_slits(tab,msa_metadata_id=1,dither_position=1):
    """ Modify the msa shutter table to expand the slits."""

    # A lot of this code was taken from assign_wcs.nirspec.get_open_msa_slits()
    
    # First we are going to filter the msa_file data on the msa_metadata_id                                                                                      
    # and dither_point_index.                                                                                                                                    
    msa_data = [np.array(x) for x in tab if x['msa_metadata_id'] == msa_metadata_id
                and x['dither_point_index'] == dither_position]

    # Get all source_ids for slitlets with sources.                                                                                                              
    # These should not be used when assigning source_id to background slitlets.                                                                                  
    source_ids = set([x[5] for x in tab if x['msa_metadata_id'] == msa_metadata_id
                      and x['dither_point_index'] == dither_position])
    #print(len(source_ids),'sources')

    
    # Get the unique slitlet_ids  
    slitlet_ids_unique = list(set([int(x['slitlet_id']) for x in msa_data]))

    # SDP may assign a value of "-1" to ``slitlet_id`` - these need to be ignored.                                                                               
    # JP-436                                                                                                                                                     
    if -1 in slitlet_ids_unique:
        slitlet_ids_unique.remove(-1)

    newtab = tab.copy()
        
    # Loop over the source_ids
    # Now lets look at each unique slitlet id                                                                                                                    
    for i,slitlet_id in enumerate(slitlet_ids_unique):
        # Get the rows for the current slitlet_id                                                                                                                
        slitlets_sid = [x for x in msa_data if x['slitlet_id'] == slitlet_id]
        open_shutters = [int(x['shutter_column']) for x in slitlets_sid]

        main_shutter = [s for s in slitlets_sid if s['primary_source'] == 'Y']
        n_main_shutter = len([s for s in slitlets_sid if s['primary_source'] == 'Y'])

        xcen, ycen, quadrant, source_xpos, source_ypos = [
                (int(s['shutter_row']), int(s['shutter_column']), int(s['shutter_quadrant']),
                 float(s['estimated_source_in_shutter_x']),
                 float(s['estimated_source_in_shutter_y']))
                for s in slitlets_sid if s['background'] == 'N'][0]
        source_id = int(main_shutter[0]['source_id'])
        
        #print(i+1,xcen,ycen,n_main_shutter,open_shutters)

        # Need three total
        if len(open_shutters):
            #needcols = [ycen-1,ycen,ycen+1]
            needcols = [ycen-2,ycen-1,ycen,ycen+1,ycen+2]
            needcols = [x for x in needcols if x not in open_shutters]
            # Loop over needed shutters
            for x in needcols:
                # slitlet_id, msa_metadata_id, quadrant, shutter_row, shutter_column, source_id, background, shutter_state,
                #   estimated_source_in_shutter_x, estimated_source_in_shutter_y, dither_point_index, primary_source
                newrow = np.array((slitlet_id, msa_metadata_id, quadrant, xcen, x, source_id, 'Y', 'OPEN', np.nan, np.nan, dither_position, 'N'))
                newtab.add_row(newrow)

    # NOTE, this does NOT check to see if there any "conflicts" of the newly added shutters with any existing shutters 
                
    return newtab

def is_binaryfile(filename):
    """ Check if a file is binary."""    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            dum = f.read()
            return False
    except UnicodeDecodeError: # Found non-text data
        return True  
