import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from dlnpyutils import utils as dln,coords
import matplotlib.pyplot as plt
import matplotlib

# each quadrant is 171 rows of 365 shutters
# the open area of the shutter is 0.20" x 0.46"  (dispersion x cross-dispersion)
# bars are 0.07" wide
# shutter pitch (center-to-center distance) is 0.27" (dispersion) and 0.53" (cross-dispersion)
# the area in each MSA quadrant is 95" (X, dispersion) by 87" (Y, cross-dispersion)
# the NIRSpec detector pixels are 0.10" x 0.10"
#
# 95" in the dispersion direction and 945 pixels, that's 95"/945 pixels = 0.10053" per pixel
# 87" in the cross-dispersion direction and 838 pixels, that's 87"/838 pixels = 0.1038"  (it's off)
# 
# so each shutter is 0.20" x 0.46" = 2.0 x 4.6 pixels
# and the bars are 0.7 pixels wide

class Line():

    def __init__(self,pars=None,x=None,width=0.7):
        self.pars = None
        self.x = None
        if pars is not None:
            self.pars = pars
        if x is not None:
            self.x = x
        self.width = width

    @property
    def slope(self):
        """ Return the slope."""
        if self.pars is None:
            return np.inf
        else:
            return self.pars[0]

    @property
    def yoffset(self):
        """ Return the Y-intercept."""
        if self.pars is None:
            return np.nan
        else:
            return self.pars[1]

    @property
    def yintercept(self):
        """ Return the Y-intercept."""
        return self.yoffset
        
    def xintercept(self,y=0):
        """ Return the X-value where the line intercepts Y."""
        if self.pars is None:
            return self.x
        else:
            # No x-intercept
            if self.slope==0:
                return np.nan
            else:
                # y = m*x + b
                # x = 1/m*(y-b)
                # when y=0
                # x = -b/m
                return (y-self.pars[1])/self.pars[0]
        
    def __call__(self,x):
        if self.pars is None:
            return x*0 + self.x
        else:
            return np.polyval(self.pars,x)

    def __repr__(self):
        prefix = self.__class__.__name__ + '('
        if self.pars is not None:
            body = 'pars=[{:.3f},{:.3f}]'.format(self.pars[0],self.pars[1])
        else:
            body = 'X={:.3f}'.format(self.x)
        out = ''.join([prefix, body, ')']) +'\n'
        return out

    def edgelines(self):
        """ Return the two edge lines."""
        # Vertical line
        if self.pars is None:
            line1 = Line(x=self.x-0.5*self.width)
            line2 = Line(x=self.x+0.5*self.width)
        # Slanted line
        else:
            # same slope
            slp = self.pars[0]
            yoff = self.pars[1]
            # the width is perpendicular to the line
            # slope = dy/dx = tan(theta)
            # sin(theta) = width/delta_yoff
            # delta_yoff = width/sin(theta)
            # also, via the hythagoream theoream
            # delta_yoff**2 = width**2 + (slp*width)**2
            halfwidth = 0.5*self.width
            delta_yoff = np.sqrt(halfwidth**2 + (slp*halfwidth)**2)
            pars1 = np.array([slp,yoff-delta_yoff])
            line1 = Line(pars=pars1)
            pars2 = np.array([slp,yoff+delta_yoff])
            line2 = Line(pars=pars2)
        return line1,line2
            
    def polygon(self,xr,yr):
        """ Get the polygon with the full width of the line and
            input xr/yr ranges."""
        xy = np.zeros(4,dtype=np.dtype([('x',float),('y',float)]))        
        # Vertical line
        if self.pars is None:
            xy['x'] = self.x + 0.5*self.width*np.array([-1,1,1,-1])
            xy['y'] = [yr[0],yr[0],yr[1],yr[1]]
        # Slanted line
        else:
            # Get edge lines
            eline1,eline2 = self.edgelines()
            # More vertical
            if np.abs(self.slope) > 1:
                # Use y-range to set edges
                xy['x'] = [eline1.xintercept(yr[0]),eline2.xintercept(yr[0]),
                           eline2.xintercept(yr[1]),eline1.xintercept(yr[1])]
                xy['y'] = [yr[0],yr[0],yr[1],yr[1]]
            # More horizontal
            else:
                # Use x-range to set edges
                xy['x'] = [xr[0],xr[1],xr[1],xr[0]]
                xy['y'] = [eline1(xr[0]),eline1(xr[1]),
                           eline2(xr[1]),eline2(xr[0])]
        return xy
        
    def coverage(self,xr,yr,osamp=1,resample=True):
        """ Return the covering fraction of a grid defined by xr/yr ranges."""
        nxpix = xr[1]-xr[0]
        nypix = yr[1]-yr[0]
        x = np.arange(nxpix*osamp)/osamp + xr[0]
        y = np.arange(nypix*osamp)/osamp + yr[0]
        im = np.zeros((nypix*osamp,nxpix*osamp),float)
        xx,yy = np.meshgrid(x,y)
        xypoly = self.polygon(xr,yr)
        # Not in range
        xmin,xmax = np.min(xypoly['x']),np.max(xypoly['x'])
        ymin,ymax = np.min(xypoly['y']),np.max(xypoly['y'])
        if xmax<xr[0] or xmin>xr[1] or ymax<yr[0] or ymin>yr[1]:
            return im
        # Only run roi_cut on the relevant portion of the grid
        _,x0 = dln.closest(x,xmin-0.1)
        _,x1 = dln.closest(x,xmax+0.1)
        _,y0 = dln.closest(y,ymin-0.1)
        _,y1 = dln.closest(y,ymax+0.1)        
        slc = (slice(y0,y1+1),slice(x0,x1+1))
        ind,cutind = dln.roi_cut(xypoly['x'],xypoly['y'],xx[slc].ravel(),yy[slc].ravel())
        if len(cutind)>0:
            cutind2 = np.unravel_index(cutind,xx[slc].shape)
            imslc = im[slc]
            imslc[cutind2[0],cutind2[1]] = 1
            im[slc] = imslc

        # Rebin
        if osamp > 1 and resample:
            im = dln.rebin(im,binsize=[osamp,osamp])
            
        return im
        
    def plot(self,xr,yr,c='r'):
        nx = xr[1]-xr[0]
        x = np.arange(nx)+xr[0]
        # Vertical line
        if self.pars is None:
            plt.plot([self.x,self.x],yr,c=c)
        else:
            plt.plot(x,self(x),c=c)
        
            
class Grid():

    def __init__(self,xcen,ycen,theta=0.0):
        # theta: positive rotates grid counter-clockwise
        self.xcen = xcen
        self.ycen = ycen
        self.theta = theta
        self.barxsep = 2.7   # pixels
        self.barysep = 5.3   # pixels
        self.barwidth = 0.7  # pixels

    def __call__(self,xr,yr,osamp=10,resample=True):
        # xr - x-range of pixel grid
        # yr - y-range of pixel grid

        # return an oversampled image
        # giving the throughput of each subpixel
        # not inclusive

        # Some negative vertical lines
        if xr[0] < self.xcen:
            nxneg = int( np.ceil((self.xcen-xr[0])/self.barxsep) )
        else:
            nxneg = 0
        # Some positive vertical lines
        if xr[1] >= self.xcen:
            nxpos = int( np.ceil((xr[1]-self.xcen)/self.barxsep) )
        else:
            nxpos = 0
        xvertical = np.arange(nxneg+nxpos+1)-nxneg

        # Some negative horizontal lines
        if yr[0] < self.ycen:
            nyneg = int( np.ceil((self.ycen-yr[0])/self.barysep) )
        else:
            nyneg = 0
        # Some positive horizontal lines
        if yr[1] >= self.ycen:
            nypos = int( np.ceil((yr[1]-self.ycen)/self.barysep) )
        yhorizontal = np.arange(nyneg+nypos+1)-nyneg        
            
        nxpix = xr[1]-xr[0]
        nypix = yr[1]-yr[0]
        x = np.arange(nxpix*osamp)/osamp + xr[0]
        y = np.arange(nypix*osamp)/osamp + yr[0]
        im = np.zeros((nypix*osamp,nxpix*osamp),float)

        # Vertical lines        
        for i,ix in enumerate(xvertical):
            vl = self.vline(ix)
            imline = vl.coverage(xr,yr,osamp=osamp,resample=False)
            im = np.maximum(im,imline)
        # Horizontal lines
        for i,iy in enumerate(yhorizontal):        
            hl = self.hline(iy)
            imline = hl.coverage(xr,yr,osamp=osamp,resample=False)
            im = np.maximum(im,imline)

        if resample:
            im = dln.rebin(im,binsize=[osamp,osamp])
        
        return im
            
    def vline(self,i):
        """ Return a vertical line with step i (+ or - integer) from the center (left or right)."""
        # theta: positive rotates grid counter-clockwise        
        x0 = self.xcen + i*self.barxsep
        if self.theta == 0:
            line = Line(x=x0)
        else:
            slp = 1/np.tan(np.deg2rad(self.theta))  # tan(theta) = dy/dx            
            # Central vertical line
            if i==0:
                # y = m*x + b
                # we know the slope, need b
                # b = y0-m*x0
                yoff = self.ycen-slp*self.xcen
                pars = np.array([slp,yoff])                
            else:
                # The vertical line intersects the central horizontal at a distance of
                # i*barxsep from the center
                dist = i*self.barxsep
                dx = np.cos(np.deg2rad(self.theta))*dist   # cos theta = opp/hyp
                dy = np.sin(np.deg2rad(self.theta))*dist   # sin theta = adj/hyp
                xintersept = self.xcen + dx
                yintersept = self.ycen + dy
                # y = m*x + b
                # we know the slope, need b
                # b = y0-m*x0
                yoff = yintersept-slp*xintersept
                pars = np.array([slp,yoff])   
            line = Line(pars=pars,width=self.barwidth)
        return line

    def hline(self,i):
        """ Return a horizontal line with step i (+ or - integer) from the center (up or down)."""
        # theta: positive rotates grid counter-clockwise        
        y0 = self.ycen + i*self.barysep
        if self.theta == 0:
            pars = np.array([0.0,y0])
        else:
            slp = np.tan(np.deg2rad(self.theta))  # tan(theta) = dy/dx                    
            # Central horizontal line
            if i==0:
                # y = m*x + b
                # we know the slope, need b
                # b = y0-m*x0
                yoff = self.ycen-slp*self.xcen
                pars = np.array([slp,yoff])
            else:
                # The horizontal line intersects the central vertical at a distance of
                # i*barysep from the center
                dist = i*self.barysep
                dy = np.cos(np.deg2rad(self.theta))*dist   # cos theta = opp/hyp
                dx = -np.sin(np.deg2rad(self.theta))*dist   # sin theta = adj/hyp
                xintersept = self.xcen + dx
                yintersept = self.ycen + dy
                # y = m*x + b
                # we know the slope, need b
                # b = y0-m*x0
                yoff = yintersept-slp*xintersept
                pars = np.array([slp,yoff])   
        line = Line(pars=pars,width=self.barwidth)     
        return line

    def plotlines(self,xr=None,yr=None):
        if xr is None:
            xr = [self.xcen-100,self.xcen+100]
        if yr is None:
            yr = [self.ycen-100,self.ycen+100]
        nxpix = xr[1]-xr[0]
        nypix = yr[1]-yr[0]
        x = np.arange(nxpix) + xr[0]
        y = np.arange(nypix) + yr[0]
        nxline = int(nxpix // self.barxsep + 1)
        if nxline % 2 == 0: nxline += 1   # must be odd
        nyline = int(nypix // self.barysep + 1)
        if nyline % 2 == 0: nyline += 1   # must be odd
        # Vertical lines
        for i in range(nxline):
            vline = self.vline(i-(nxline//2))
            vline.plot(xr,yr)
        # Horizontal lines
        for i in range(nyline):
            hline = self.hline(i-(nyline//2))
            hline.plot(xr,yr)
        
        
def model_background():
    """ Model a NIRSPEC MSATA image with just the background."""

    # represent each bar as a line with a certain thickness

    # oversample each pixel by 10 x 10 
    
    import pdb; pdb.set_trace()


    
def model_star():
    """ Model a NIRSPEC MSATA image with a star and background."""
    
    pass
