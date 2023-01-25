import numpy as np
import matplotlib.pyplot as plt


def slitsim(x0,y0,plot=False):
    """
    Simulate a star in the shutter
    """

    # in pixels units,
    # shutter is 2 x 5 pixels
    x = np.arange(-6,6,0.01)
    nx = len(x)
    y = np.arange(-10,10,0.01)
    ny = len(y)
    xx,yy = np.meshgrid(x,y)
    fwhm = 0.9
    sigma = fwhm/2.35
    x0 = -1.0 #-0.25 #0.0
    y0 = 0.0
    psf = np.exp(-0.5*((xx-x0)**2+(yy-y0)**2)/sigma**2)
    psf /= np.sum(psf)

    # shutter
    good = ((np.abs(xx)<=1) & (np.abs(yy)<=2.5))
    print('flux fraction=',np.sum(psf[good])/np.sum(psf))

    psf_masked = psf.copy()
    psf_masked[~good]=0

    xcen = np.sum(psf_masked*xx)/np.sum(psf_masked)
    ycen = np.sum(psf_masked*yy)/np.sum(psf_masked)
    print('central position=',x0,y0)
    print('boxed position=',xcen,ycen)
    print('offset=',xcen-x0,ycen-y0)

    if plot:
        pl.display(psf_masked,x,y)
        pl.oplot([-1,1,1,-1,-1],[-2.5,-2.5,2.5,2.5,-2.5],c='r')
        plt.scatter([x0],[y0],c='r',marker='+')
        plt.xlim(-2,2)
        plt.ylim(-3,3)
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.scatter([xcen],[ycen],marker='D',facecolor='none',edgecolor='green')

def synthstar(pars,snr=100,withwiggles=True):
    """
    Create synthetic/fake star from the FERRE grid
    """

    
