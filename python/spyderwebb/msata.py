import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from dlnpyutils import utils as dln,coords
from scipy.optimize import curve_fit
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


class Box():

    def __init__(self,xcen=0,ycen=0,width=2.0,height=4.7,theta=0.0):
        self.xcen = xcen
        self.ycen = ycen
        self.width = width
        self.height = height
        self.theta = theta
        # theta: positive rotates counter-clockwise
        
    def __call__(self,xr=None,yr=None):
        """ Return the model."""
        pass

    def __repr__(self):
        prefix = self.__class__.__name__ + '('
        body = 'Xc={:.2f},Yc={:.2f},Width={:.2f}'.format(self.xcen,self.ycen,self.width)
        body += ',Height={:.2f},Theta={:.2f}'.format(self.height,self.theta)
        out = ''.join([prefix, body, ')']) +'\n'
        return out

    @property
    def center(self):
        return self.xcen,self.ycen

    @property
    def x(self):
        """ Return polygon X-values"""        
        return self.polygon()['x']

    @property
    def y(self):
        """ Return polygon Y-values"""
        return self.polygon()['y']    
    
    @property
    def xrange(self):
        """ Return the X-range."""
        xy = self.polygon()
        return np.min(xy['x']),np.max(xy['x'])

    @property
    def yrange(self):
        """ Return the Y-range."""
        xy = self.polygon()
        return np.min(xy['y']),np.max(xy['y'])    
    
    def polygon(self):
        """ Return the points of the polygon."""
        xy = np.zeros(4,dtype=np.dtype([('x',float),('y',float)]))  
        if self.theta==0:
            xy['x'] = np.array([-1,1,1,-1])*0.5*self.width+self.xcen
            xy['y'] = np.array([-1,-1,1,1])*0.5*self.height+self.ycen
        else:
            x0 = np.array([-1,1,1,-1])*0.5*self.width
            y0 = np.array([-1,-1,1,1])*0.5*self.height
            sth = np.sin(np.deg2rad(self.theta))
            cth = np.cos(np.deg2rad(self.theta))            
            #rot = np.array([[cth,-sth],
            #                [sth,cth]])
            x = x0*cth - y0*sth
            y = x0*sth + y0*cth
            xy['x'] = x + self.xcen
            xy['y'] = y + self.ycen
        return xy

    def coverage(self,xr=None,yr=None,osamp=1,resample=True):
        """ Return the covering fraction of a grid defined by xr/yr ranges."""
        if xr is None:
            xr = self.xrange
        if yr is None:
            yr = self.yrange
        nxpix = xr[1]-xr[0]
        nypix = yr[1]-yr[0]
        x = np.arange(nxpix*osamp)/osamp + xr[0]
        y = np.arange(nypix*osamp)/osamp + yr[0]
        im = np.zeros((nypix*osamp,nxpix*osamp),float)
        xx,yy = np.meshgrid(x,y)
        xypoly = self.polygon()
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
    
    def plot(self,c='r',filled=False):
        xy = self.polygon()
        if filled:
            plt.fill(np.append(xy['x'],xy['x'][0]),
                     np.append(xy['y'],xy['y'][0]),c=c)
        else:
            plt.plot(np.append(xy['x'],xy['x'][0]),
                     np.append(xy['y'],xy['y'][0]),c=c)
    
            
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

def check_shutter_working(quad,im1,im2):
    """ Check if the shutters are working in the input images."""

    quad['offdetector'] = False
    quad['working'] = False
    quad['brokenclosed'] = False
    quad['brokenopen'] = False    
    for i in range(len(quad)):
        # off the detector
        if quad['x'][i]<0 or quad['x'][i]>2047 or quad['y'][i]<0 or quad['y'][i]>2047:
            quad['offdetector'][i] = True
            continue
        xlo = int(np.round(quad['x'][i]-1))
        xlo = np.maximum(xlo,0)
        xhi = int(np.round(quad['x'][i]+2))
        xhi = np.minimum(xhi,2048)
        ylo = int(np.round(quad['y'][i]-2))
        ylo = np.maximum(ylo,0)
        yhi = int(np.round(quad['y'][i]+3))
        yhi = np.minimum(yhi,2048)        
        sim1 = im1[ylo:yhi,xlo:xhi]
        sim2 = im2[ylo:yhi,xlo:xhi]        
        sum1 = np.nansum(np.maximum(sim1,0))
        sum2 = np.nansum(np.maximum(sim2,0))
        med1 = np.nanmedian(sim1)
        med2 = np.nanmedian(sim2)        
        max1 = np.nanmax(sim1)
        max2 = np.nanmax(sim2)
        nbad = np.sum(~np.isfinite(sim1))
        # sum isn't great because there can be bad pixels, and we can catch the edge of the neighboring shutter
        # max is not great because it is sensitive to outlier pixels
        
        #detect1 = (sum1>100) and (max1>50)
        #detect2 = (sum2>100) and (max2>50)
        detect1 = med1>40
        detect2 = med2>40
        if (detect1==True and detect2==False) or (detect1==False and detect2==True):
            quad['working'][i] = True
            cmt = 'working'
        elif (detect1==False and detect2==False):
            quad['brokenclosed'][i] = True
            cmt = 'brokenclosed'
        elif (detect1==True and detect2==True):
            quad['brokenopen'][i] = True
            cmt = 'brokenopen'

        if nbad>5:
            quad['badpixels'][i] = True
            
        print(i,quad['col'][i],quad['row'][i],quad['x'][i],quad['y'][i],sum1,sum2,med1,med2,max1,max2,detect1,detect2,nbad,cmt)

        #if i==10224 or i==9859:
        #    import pdb; pdb.set_trace()
        
        #if sum1>100 or sum2>100:
        #    import pdb; pdb.set_trace()

    print('total = ',len(quad))
    print('offdetector = ',np.sum(quad['offdetector']==True))
    print('working = ',np.sum(quad['working']==True))    
    print('brokenclosed = ',np.sum(quad['brokenclosed']==True))
    print('brokenopen = ',np.sum(quad['brokenopen']==True))

    # medthresh 50
    # total =  62415
    # offdetector =  730
    # working =  47979
    # brokenclosed =  13342
    # brokenopen =  364

    # medthresh 20 
    # total =  62415
    # offdetector =  730
    # working =  30591
    # brokenclosed =  12482
    # brokenopen =  18612
    
    
    #import pdb; pdb.set_trace()

    return quad
            
# The positions of all the shutters
def make_shutter_table():

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


    # ========================== NRS1 ===============================    
    hdu1 = fits.open('jw04461052001_01201_00001_nrs1_rate.fits.gz')
    hdu2 = fits.open('jw04461053001_01201_00001_nrs1_rate.fits.gz')
    im1 = hdu1[1].data
    im2 = hdu2[1].data
    im = im1+im2
    #mask = np.zeros(im.shape,bool)
    #mask[1204:2045,1063:2010] = True   # Q3
    #mask[4:854,1059:2013] = True       # Q4
    
    # Each quadrant is 171 rows of 365 shutters

    ################# Q4 #######################
    
    # The first (bottom) row is half cutoff
    # (x,y)= (1060.33,3.4)
    # the first 4 columns are cut off at the top
    # the last shutter of the first column has a position of
    # (x,y) = (1061.8,326.8)
    # The first shutter of the 5th column is
    # (x,y) = (1071.06,3.4)
    # the last shutter of the 5th column is
    # (x,y) = (174.0,850.0)
    # plt.plot([1071,1074.2],[3.5,850],c='r')
    # vertical separation is 5.09 pixels

    # first full column is

    # 850.0
    # 703.5
    # 29 shutters separating them
    # 5.05

    # 169 rows (visible), two rows missing off the bottom
    #y = np.arange(169)*5.041+3.4
    #x = np.arange(169)*0.0175+1071.06
    y1 = np.arange(171)*5.041-6.682
    x1 = np.arange(171)*0.0175+1071.0250
    c1 = np.zeros(171)+4
    r1 = np.arange(171)
    
    # (near) last column (column index=290)
    y2 = np.arange(171)*5.040-6.102
    x2 = np.arange(171)*0.002+1818.27
    c2 = np.zeros(171)+290
    r2 = np.arange(171)
    
    pl.display(im2,vmin=0,vmax=500,xr=[1050,2000],yr=[0,900])
    plt.scatter(x,y,marker='+',c='r')
    plt.xlim(1800,1840)
    plt.ylim(750,800)

    # the columns
    # bottom row (row index = 3)
    # 1080.84 - 1060.2 8 steps, 2.58 pixels
    x3 = np.arange(365)*2.613+1060.50
    y3 = np.arange(365)*0.002+8.3
    c3 = np.arange(365)
    r3 = np.zeros(365) + 3
    
    # top row (row index = 170)
    x4 = np.arange(365)*2.603+1064.0
    y4 = np.arange(365)*0.0018+850.288
    c4 = np.arange(365)
    r4 = np.zeros(365)+170
    
    pl.display(im2,vmin=0,vmax=500,xr=[1050,2048],yr=[0,860])
    plt.scatter(x,y,marker='+',c='r')
    plt.xlim(1850,1900)

    # fit x as functions of col/row index
    # only use bottom/top rows,
    allx = np.concatenate((x3,x4))
    ally = np.concatenate((y3,y4))
    allc = np.concatenate((c3,c4))
    allr = np.concatenate((r3,r4))
    # do the fitting
    estimates = np.zeros(4)
    xpars,xcov = curve_fit(dln.poly2d_wrap,[allc,allr],allx,p0=estimates)
    # xpars = np.array([ 1.06043713e+03,  2.61317964e+00, -5.98802395e-05,  2.09580838e-02])
    xbest = dln.poly2d_wrap([allc,allr],*xpars)
    rms = np.sqrt(np.mean((allx-xbest)**2))
    print('rms = ',rms)
    pl.scatter(allc,allr,allx-xbest)

    # fit y as functions of col/row index
    # only use left/right columns
    allx = np.concatenate((x1,x2))
    ally = np.concatenate((y1,y2))
    allc = np.concatenate((c1,c2))
    allr = np.concatenate((r1,r2))
    # do the fitting
    estimates = np.zeros(4)
    ypars,ycov = curve_fit(dln.poly2d_wrap,[allc,allr],ally,p0=estimates)
    # ypars = np.array([-6.69011189e+00,  2.02797203e-03, -3.49650350e-06,  5.04101399e+00])
    ybest = dln.poly2d_wrap([allc,allr],*ypars)
    rms = np.sqrt(np.mean((ally-ybest)**2))
    print('rms = ',rms)
    pl.scatter(allc,allr,ally-ybest)
    
    
    # full quadrant positions
    col,row = np.meshgrid(np.arange(365),np.arange(171))
    # Each coefficient is also a linear equation
    #xx = col*(2.6132-6.00e-5*row) + row*(0.0175-5.344e-5*col) + 1060.4475
    #yy = col*(0.002-1.17e-6*row) + row*(5.0400+3.45e-6*col) - 6.682
    xx = dln.poly2d(col,row,*xpars)
    yy = dln.poly2d(col,row,*ypars)    
    
    pl.display(im1,vmin=0,vmax=500,xr=[1050,2048],yr=[0,860])
    plt.scatter(xx,yy,marker='+',c='r')
    #plt.xlim(1750,1800)

    # Make the table
    dt = [('nrs',int),('quad',int),('row',int),('col',int),('x',float),('y',float),
          ('offdetector',bool),('badpixels',bool),('working',bool),('brokenopen',bool),('brokenclosed',bool)]
    quad4 = np.zeros(171*365,dtype=np.dtype(dt))
    quad4['nrs'] = 1
    quad4['quad'] = 4
    quad4['row'] = row.ravel()
    quad4['col'] = col.ravel()
    quad4['x'] = xx.ravel()
    quad4['y'] = yy.ravel()


    # Figure out which ones are working or broken
    out = msata.check_shutter_working(quad4,im1,im2)
    col = np.zeros(len(out),int)
    col[out['offdetector']] = 1
    col[out['working']] = 2
    col[out['brokenopen']] = 3
    col[out['brokenclosed']] = 4
    # Table(out).write('nirspec_msata_quad4.fits')

    plt.figure(1)
    plt.clf()
    plt.imshow(im1,origin='lower',aspect='auto',vmin=0,vmax=500,interpolation='none',cmap='Greys_r')
    o=plt.scatter(out['x'],out['y'],c=col,marker='+',cmap='jet')
    plt.colorbar(o)
    #plt.xlim(1055,1092)
    #plt.ylim(653,696)

    plt.figure(2)
    plt.clf()
    plt.imshow(im2,origin='lower',aspect='auto',vmin=0,vmax=500,interpolation='none',cmap='Greys_r')
    o=plt.scatter(out['x'],out['y'],c=col,marker='+',cmap='jet')
    plt.colorbar(o)
    #plt.xlim(1055,1092)
    #plt.ylim(653,696)

    

    ################# Q3 #######################

    # X: [1050,2017]
    # Y: [1190,2048]

    # first column (column index=0)
    # (x,y)=(1065.0,1205.1)
    
    # vertical steps
    # 1492.5 - 1402.3, 18 steps
    # 5.011 pixels

    y1 = np.arange(171)*4.998+1206.8
    x1 = np.arange(171)*0.019+1065.3
    c1 = np.zeros(171) + 0
    r1 = np.arange(171)

    pl.display(im1,vmin=0,vmax=500,xr=[1060,1075],yr=[1160,2048])
    plt.scatter(x1,y1,marker='+',c='r')
    plt.ylim(1224,1242)
    #plt.ylim(1350,1400)    

    # last column (column index=363)
    # (x,y)=(2008.6,1206.8)
    # the very last column is completely missing
    
    # vertical steps
    # 1342.5 - 1212.1, 26 steps
    # 5.0154 pixels
    
    y2 = np.arange(171)*4.999+1206.8
    x2 = np.arange(171)*0.0005+2008.6
    c2 = np.zeros(171)+363
    r2 = np.arange(171)
    
    pl.display(im2,vmin=0,vmax=500,xr=[1050,2048],yr=[1160,2048])
    plt.scatter(x2,y2,marker='+',c='r')
    plt.xlim(1990,2018)
    plt.ylim(1180,1300)

    # the columns

    # I only see 364 columns, either the first or the last one is completely missing
    # by looking at the bottom quadrant, it's clear that the last column is missing
    
    # bottom row (row index = 0)
    # horizontal steps
    # (1156.56-1078.5)/30
    # 2.602 pixels
    x3 = np.arange(365)*2.598+1065.5
    y3 = np.arange(365)*0.0003+1206.8
    c3 = np.arange(365)
    r3 = np.zeros(365) + 0

    pl.display(im2,vmin=0,vmax=500,xr=[1050,2048],yr=[1200,2048])
    plt.scatter(x3,y3,marker='+',c='r')
    plt.ylim(1200,1250)
    plt.xlim(1900,1950)
    
    # (near) top row (row index = 166)
    #  the top rows are cut off
    x4 = np.arange(365)*2.5875+1068.9
    y4 = np.arange(365)*0.0000+2036.8
    c4 = np.arange(365)
    r4 = np.zeros(365) + 166
    
    pl.display(im2,vmin=0,vmax=500,xr=[1050,2048],yr=[2020,2048])
    plt.scatter(x4,y4,marker='+',c='r')
    plt.xlim(1750,1800)

    # fit x as functions of col/row index
    # only use bottom/top rows,
    allx = np.concatenate((x3,x4))
    ally = np.concatenate((y3,y4))
    allc = np.concatenate((c3,c4))
    allr = np.concatenate((r3,r4))
    # do the fitting
    estimates = np.zeros(4)
    xpars,xcov = curve_fit(dln.poly2d_wrap,[allc,allr],allx,p0=estimates)
    # xpars = np.array([ 1.06550000e+03,  2.59800000e+00, -6.32530120e-05,  2.04819277e-02])
    xbest = dln.poly2d_wrap([allc,allr],*xpars)
    rms = np.sqrt(np.mean((allx-xbest)**2))
    print('rms = ',rms)
    pl.scatter(allc,allr,allx-xbest)

    # fit y as functions of col/row index
    # only use left/right columns
    allx = np.concatenate((x1,x2))
    ally = np.concatenate((y1,y2))
    allc = np.concatenate((c1,c2))
    allr = np.concatenate((r1,r2))
    # do the fitting
    estimates = np.zeros(4)
    ypars,ycov = curve_fit(dln.poly2d_wrap,[allc,allr],ally,p0=estimates)
    # ypars = np.array([1.20680000e+03, 5.40635704e-09, 2.75478556e-06, 4.99800001e+00])
    ybest = dln.poly2d_wrap([allc,allr],*ypars)
    rms = np.sqrt(np.mean((ally-ybest)**2))
    print('rms = ',rms)
    pl.scatter(allc,allr,ally-ybest)
    
    
    # full quadrant positions
    col,row = np.meshgrid(np.arange(365),np.arange(171))
    # Each coefficient is also a linear equation
    #xx = col*(2.6132-6.00e-5*row) + row*(0.0175-5.344e-5*col) + 1060.4475
    #yy = col*(0.002-1.17e-6*row) + row*(5.0400+3.45e-6*col) - 6.682
    xx = dln.poly2d(col,row,*xpars)
    yy = dln.poly2d(col,row,*ypars)    
    
    pl.display(im1,vmin=0,vmax=500,xr=[1050,2048],yr=[1050,2048])
    plt.scatter(xx,yy,marker='+',c='r')
    #plt.xlim(1750,1800)

    # Make the table
    dt = [('nrs',int),('quad',int),('row',int),('col',int),('x',float),('y',float),
          ('offdetector',bool),('badpixels',bool),('working',bool),('brokenopen',bool),('brokenclosed',bool)]
    quad3 = np.zeros(171*365,dtype=np.dtype(dt))
    quad3['nrs'] = 1
    quad3['quad'] = 3
    quad3['row'] = row.ravel()
    quad3['col'] = col.ravel()
    quad3['x'] = xx.ravel()
    quad3['y'] = yy.ravel()


    # Figure out which ones are working or broken
    out = msata.check_shutter_working(quad3,im1,im2)
    col = np.zeros(len(out),int)
    col[out['offdetector']] = 1
    col[out['working']] = 2
    col[out['brokenopen']] = 3
    col[out['brokenclosed']] = 4
    # Table(out).write('nirspec_msata_quad3.fits')

    plt.figure(1)
    plt.clf()
    plt.imshow(im1,origin='lower',aspect='auto',vmin=0,vmax=500,interpolation='none',cmap='Greys_r')
    o=plt.scatter(out['x'],out['y'],c=col,marker='+',cmap='jet')
    plt.colorbar(o)
    #plt.xlim(1055,1092)
    #plt.ylim(653,696)

    plt.figure(2)
    plt.clf()
    plt.imshow(im2,origin='lower',aspect='auto',vmin=0,vmax=500,interpolation='none',cmap='Greys_r')
    o=plt.scatter(out['x'],out['y'],c=col,marker='+',cmap='jet')
    plt.colorbar(o)
    #plt.xlim(1055,1092)
    #plt.ylim(653,696)



    
    # ========================== NRS2 ===============================
    hdu1 = fits.open('jw04461052001_01201_00001_nrs2_rate.fits.gz')
    hdu2 = fits.open('jw04461053001_01201_00001_nrs2_rate.fits.gz')
    orig = hdu1[1].data + hdu2[1].data
    im1 = hdu1[1].data
    im2 = hdu2[1].data
    im = im1 + im2
    #mask = np.zeros(im.shape,bool)
    #mask[1205:2045,40:988] = True    # Q1
    #mask[4:854,45:1000] = True       # Q2

    # Each quadrant is 171 rows of 365 shutters

    ################# Q2 #######################

    # X: [45,1000]
    # Y: [4,854]
    
    # first full column is
    # (x,y)=(47.7,8.5)

    # 169 rows (visible), two rows missing off the bottom
    y1 = np.arange(171)*5.040-6.02   #+9.1
    x1 = np.arange(171)*-0.007+47.5
    c1 = np.zeros(171) + 0
    r1 = np.arange(171)

    pl.display(im2,vmin=0,vmax=500,xr=[0,1050],yr=[0,900])
    plt.scatter(x1,y1,marker='+',c='r')
    plt.xlim(35,70)
    plt.ylim(800,850)

    
    # last column (column index=364)
    y2 = np.arange(171)*5.040-6.102
    x2 = np.arange(171)*-0.031+998.1
    c2 = np.zeros(171)+364
    r2 = np.arange(171)
    
    pl.display(im2,vmin=0,vmax=500,xr=[0,1000],yr=[0,900])
    plt.scatter(x2,y2,marker='+',c='r')
    plt.xlim(980,1005)
    plt.ylim(700,750)

    
    # the columns
    
    # (near) bottom row (row index = 3)
    x3 = np.arange(365)*2.6112+47.5
    y3 = np.arange(365)*0.0001+9.03
    c3 = np.arange(365)
    r3 = np.zeros(365) + 3

    pl.display(im2,vmin=0,vmax=500,xr=[0,1000],yr=[0,900])
    plt.scatter(x3,y3,marker='+',c='r')
    plt.ylim(5,20)
    plt.xlim(850,900)    

    # (near) top row (row index = 167)
    x4 = np.arange(365)*2.6010+46.331
    y4 = np.arange(365)*0.0002+835.66
    c4 = np.arange(365)
    r4 = np.zeros(365)+167
    
    pl.display(im2,vmin=0,vmax=500,xr=[0,1000],yr=[0,900])
    plt.scatter(x4,y4,marker='+',c='r')
    plt.ylim(830,860)
    plt.xlim(950,1000)

    # fit x as functions of col/row index
    # only use bottom/top rows,
    allx = np.concatenate((x3,x4))
    ally = np.concatenate((y3,y4))
    allc = np.concatenate((c3,c4))
    allr = np.concatenate((r3,r4))
    # do the fitting
    estimates = np.zeros(4)
    xpars,xcov = curve_fit(dln.poly2d_wrap,[allc,allr],allx,p0=estimates)
    # xpars = np.array([ 4.75213841e+01,  2.61138659e+00, -6.21951220e-05, -7.12804878e-03])
    xbest = dln.poly2d_wrap([allc,allr],*xpars)
    rms = np.sqrt(np.mean((allx-xbest)**2))
    print('rms = ',rms)
    pl.scatter(allc,allr,allx-xbest)

    # fit y as functions of col/row index
    # only use left/right columns
    allx = np.concatenate((x1,x2))
    ally = np.concatenate((y1,y2))
    allc = np.concatenate((c1,c2))
    allr = np.concatenate((r1,r2))
    # do the fitting
    estimates = np.zeros(4)
    ypars,ycov = curve_fit(dln.poly2d_wrap,[allc,allr],ally,p0=estimates)
    # ypars = np.array([-6.01999965e+00, -2.82761004e-04,  2.80250461e-11,  5.04000000e+00])
    ybest = dln.poly2d_wrap([allc,allr],*ypars)
    rms = np.sqrt(np.mean((ally-ybest)**2))
    print('rms = ',rms)
    pl.scatter(allc,allr,ally-ybest)
    
    
    # full quadrant positions
    col,row = np.meshgrid(np.arange(365),np.arange(171))
    # Each coefficient is also a linear equation
    #xx = col*(2.6132-6.00e-5*row) + row*(0.0175-5.344e-5*col) + 1060.4475
    #yy = col*(0.002-1.17e-6*row) + row*(5.0400+3.45e-6*col) - 6.682
    xx = dln.poly2d(col,row,*xpars)
    yy = dln.poly2d(col,row,*ypars)    
    
    pl.display(im1,vmin=0,vmax=500,xr=[0,1000],yr=[0,900])
    plt.scatter(xx,yy,marker='+',c='r')
    #plt.xlim(1750,1800)

    # Make the table
    dt = [('nrs',int),('quad',int),('row',int),('col',int),('x',float),('y',float),
          ('offdetector',bool),('badpixels',bool),('working',bool),('brokenopen',bool),('brokenclosed',bool)]
    quad2 = np.zeros(171*365,dtype=np.dtype(dt))
    quad2['nrs'] = 1
    quad2['quad'] = 2
    quad2['row'] = row.ravel()
    quad2['col'] = col.ravel()
    quad2['x'] = xx.ravel()
    quad2['y'] = yy.ravel()


    # Figure out which ones are working or broken
    out = msata.check_shutter_working(quad2,im1,im2)
    col = np.zeros(len(out),int)
    col[out['offdetector']] = 1
    col[out['working']] = 2
    col[out['brokenopen']] = 3
    col[out['brokenclosed']] = 4
    # Table(out).write('nirspec_msata_quad2.fits')

    plt.figure(1)
    plt.clf()
    plt.imshow(im1,origin='lower',aspect='auto',vmin=0,vmax=500,interpolation='none',cmap='Greys_r')
    o=plt.scatter(out['x'],out['y'],c=col,marker='+',cmap='jet')
    plt.colorbar(o)
    #plt.xlim(1055,1092)
    #plt.ylim(653,696)

    plt.figure(2)
    plt.clf()
    plt.imshow(im2,origin='lower',aspect='auto',vmin=0,vmax=500,interpolation='none',cmap='Greys_r')
    o=plt.scatter(out['x'],out['y'],c=col,marker='+',cmap='jet')
    plt.colorbar(o)
    #plt.xlim(1055,1092)
    #plt.ylim(653,696)

    # Each quadrant is 171 rows of 365 shutters

    ################# Q1 #######################
    

    # X: [0,1020]
    # Y: [1200,2048]

    # first column (column index=0)
    # (x,y)=(42.7,1206.85)
    
    # vertical steps
    # 1492.5 - 1402.3, 18 steps
    # 5.011 pixels

    y1 = np.arange(171)*4.997+1207.5
    x1 = np.arange(171)*-0.0078+42.5
    c1 = np.zeros(171) + 0
    r1 = np.arange(171)

    pl.display(im1,vmin=0,vmax=500,xr=[0,1000],yr=[1160,2048])
    plt.scatter(x1,y1,marker='+',c='r')
    plt.xlim(35,50)
    plt.ylim(2000,2048)    

    # (near) last column (column index=363)
    # (x,y)=(985.13,1207.06)

    y2 = np.arange(171)*4.996+1207.4
    x2 = np.arange(171)*-0.029+984.9
    c2 = np.zeros(171)+363
    r2 = np.arange(171)
    
    pl.display(im2,vmin=0,vmax=500,xr=[0,1000],yr=[1200,2048])
    plt.scatter(x2,y2,marker='+',c='r')
    plt.xlim(970,1000)
    plt.ylim(2000,2048)

    # the columns

    # I only see 364 columns, either the first or the last one is completely missing
    # by looking at the bottom quadrant, it's clear that the last column is missing
    
    # bottom row (row index = 0)
    # (x,y)=(42.37,1207.25)
    # horizontal steps
    # (1156.56-1078.5)/30
    # 2.602 pixels
    x3 = np.arange(365)*2.597+42.3
    y3 = np.arange(365)*0.0003+1207.4
    c3 = np.arange(365)
    r3 = np.zeros(365) + 0

    pl.display(im2,vmin=0,vmax=500,xr=[0,1100],yr=[1100,2048])
    plt.scatter(x3,y3,marker='+',c='r')
    plt.ylim(1200,1250)
    plt.xlim(900,950)
    
    # (near) top row (row index = 165)
    #  the top rows are cut off
    x4 = np.arange(365)*2.5863+41.3
    y4 = np.arange(365)*-0.001+2031.9
    c4 = np.arange(365)
    r4 = np.zeros(365) + 165
    
    pl.display(im2,vmin=0,vmax=500,xr=[0,1000],yr=[1100,2048])
    plt.scatter(x4,y4,marker='+',c='r')
    plt.ylim(2020,2040)
    plt.xlim(950,1000)


    # fit x as functions of col/row index
    # only use bottom/top rows,
    allx = np.concatenate((x3,x4))
    ally = np.concatenate((y3,y4))
    allc = np.concatenate((c3,c4))
    allr = np.concatenate((r3,r4))
    # do the fitting
    estimates = np.zeros(4)
    xpars,xcov = curve_fit(dln.poly2d_wrap,[allc,allr],allx,p0=estimates)
    # xpars = np.array([ 4.23000000e+01,  2.59700000e+00, -6.48484848e-05, -6.06060606e-03])
    xbest = dln.poly2d_wrap([allc,allr],*xpars)
    rms = np.sqrt(np.mean((allx-xbest)**2))
    print('rms = ',rms)
    pl.scatter(allc,allr,allx-xbest)

    # fit y as functions of col/row index
    # only use left/right columns
    allx = np.concatenate((x1,x2))
    ally = np.concatenate((y1,y2))
    allc = np.concatenate((c1,c2))
    allr = np.concatenate((r1,r2))
    # do the fitting
    estimates = np.zeros(4)
    ypars,ycov = curve_fit(dln.poly2d_wrap,[allc,allr],ally,p0=estimates)
    # ypars = np.array([ 1.20750000e+03, -2.75482098e-04, -2.75482092e-06,  4.99700000e+00])
    ybest = dln.poly2d_wrap([allc,allr],*ypars)
    rms = np.sqrt(np.mean((ally-ybest)**2))
    print('rms = ',rms)
    pl.scatter(allc,allr,ally-ybest)
    
    
    # full quadrant positions
    col,row = np.meshgrid(np.arange(365),np.arange(171))
    # Each coefficient is also a linear equation
    #xx = col*(2.6132-6.00e-5*row) + row*(0.0175-5.344e-5*col) + 1060.4475
    #yy = col*(0.002-1.17e-6*row) + row*(5.0400+3.45e-6*col) - 6.682
    xx = dln.poly2d(col,row,*xpars)
    yy = dln.poly2d(col,row,*ypars)    
    
    pl.display(im1,vmin=0,vmax=500,xr=[0,1100],yr=[1050,2048])
    plt.scatter(xx,yy,marker='+',c='r')
    #plt.xlim(1750,1800)

    # Make the table
    dt = [('nrs',int),('quad',int),('row',int),('col',int),('x',float),('y',float),
          ('offdetector',bool),('badpixels',bool),('working',bool),('brokenopen',bool),('brokenclosed',bool)]
    quad1 = np.zeros(171*365,dtype=np.dtype(dt))
    quad1['nrs'] = 1
    quad1['quad'] = 1
    quad1['row'] = row.ravel()
    quad1['col'] = col.ravel()
    quad1['x'] = xx.ravel()
    quad1['y'] = yy.ravel()


    # Figure out which ones are working or broken
    out = msata.check_shutter_working(quad1,im1,im2)
    col = np.zeros(len(out),int)
    col[out['offdetector']] = 1
    col[out['working']] = 2
    col[out['brokenopen']] = 3
    col[out['brokenclosed']] = 4
    # Table(out).write('nirspec_msata_quad1.fits')

    plt.figure(1)
    plt.clf()
    plt.imshow(im1,origin='lower',aspect='auto',vmin=0,vmax=500,interpolation='none',cmap='Greys_r')
    o=plt.scatter(out['x'],out['y'],c=col,marker='+',cmap='jet')
    plt.colorbar(o)
    #plt.xlim(1055,1092)
    #plt.ylim(653,696)

    plt.figure(2)
    plt.clf()
    plt.imshow(im2,origin='lower',aspect='auto',vmin=0,vmax=500,interpolation='none',cmap='Greys_r')
    o=plt.scatter(out['x'],out['y'],c=col,marker='+',cmap='jet')
    plt.colorbar(o)
    #plt.xlim(1055,1092)
    #plt.ylim(653,696)

    from astropy.table import vstack,hstack
    quad1 = Table.read('nirspec_msata_quad1.fits')
    quad2 = Table.read('nirspec_msata_quad2.fits')
    quad3 = Table.read('nirspec_msata_quad3.fits')
    quad4 = Table.read('nirspec_msata_quad4.fits')    
    quad = vstack((quad1,quad2,quad3,quad4))
    quad.write('nirspec_msa_quadrants.fits')

    # not fit each shutter better using oversampled modeling
    # us the new Box() class
    

def model_background(im,verbose=True):
    """ Model a NIRSPEC MSATA image with just the background."""

    med = np.nanmedian(im)
    sig = dln.mad(im)
    ny,nx = im.shape
    xcen = nx//2
    ycen = ny//2
    xr = [0,nx]
    yr = [0,ny]
    
    mask = (np.abs(im-med) < 3*sig) & np.isfinite(im)
    err = np.sqrt(im)
    err[~np.isfinite(err)] = np.nanmedian(err)

    
    def msatafitter(x,*pars):
        if verbose:
            print(pars)
        amp = pars[0]
        xcen = pars[1]
        ycen = pars[2]
        if len(pars)>3:
            theta = pars[3]
        else:
            theta = 0.0
        if len(pars)>4:
            barxsep = pars[4]
            barysep = pars[5]
            barwidth = pars[6]
        grid = Grid(xcen,ycen,theta=theta)

        if len(pars)>4:
            grid.barxsep = barxsep
            grid.barysep = barysep
            grid.barwidth = barwidth
        model = amp*(1-grid(xr,yr))
        return model[mask]

    def msatajac(x,*pars):
        jac = np.zeros((len(x),len(pars)),float)
        f0 = msatafitter(x,*pars)
        for i in range(len(pars)):
            tpars = np.array(pars).copy()
            dx = tpars[i]*0.20
            if dx==0:
                dx = 0.01
            tpars[i] += dx
            f1 = msatafitter(x,*tpars)
            jac[:,i] = (f1-f0)/dx
        return jac

    # First only fit [amp, xcen, ycen, theta]
    estimates = [med, xcen, ycen, 0.0]
    print('estimates = ',estimates)
    bounds = [np.zeros(len(estimates))-np.inf,np.zeros(len(estimates))+np.inf]
    bounds[0][0] = 0
    bounds[0][1] = xcen-5
    bounds[1][1] = xcen+5
    bounds[0][2] = ycen-5
    bounds[1][2] = ycen+5
    pars1,pcov1 = curve_fit(msatafitter,im[mask]*0,im[mask],sigma=err[mask],p0=estimates,bounds=bounds,jac=msatajac)
    print('pars = ',pars1)

    grid = Grid(pars1[1],pars1[2],theta=0.0)
    bestmodel = pars1[0]*(1-grid(xr,yr))
    chisq = np.sum((im[mask]-bestmodel[mask])**2 / err[mask]**2)/np.sum(mask)
    print('chisq = ',chisq)
    
    import pdb; pdb.set_trace()
     
    # [amp, xce, ycen, theta, barxsep, barysep, barwidth]
    estimates = [pars1[0], pars1[1], pars1[2], pars1[3], 2.7, 5.3, 0.7]
    bounds = [np.zeros(len(estimates))-np.inf,np.zeros(len(estimates))+np.inf]
    bounds[0][0] = 0
    bounds[0][1] = estimates[1]-3
    bounds[1][1] = estimates[1]+3
    bounds[0][2] = estimates[2]-3
    bounds[1][2] = estimates[2]+3
    bounds[0][4] = 2.5    # barxsep
    bounds[1][4] = 3.0
    bounds[0][5] = 5.0    # barysep
    bounds[1][5] = 5.6
    bounds[0][6] = 0.5    # barwidth
    bounds[1][6] = 0.9
    
    pars,pcov = curve_fit(msatafitter,im[mask]*0,im[mask],sigma=err[mask],p0=estimates,bounds=bounds,jac=msatajac)
    
    print('best pars: ',pars)
    grid = Grid(pars[1],pars[2],theta=pars[3])
    grid.barxsep = pars[4]
    grid.barysep = pars[5]
    grid.barwidth = pars[6]
    bestmodel = pars[0]*(1-grid(xr,yr))
    chisq = np.sum((im[mask]-bestmodel[mask])**2 / err[mask]**2)/np.sum(mask)    
    print('chisq = ',chisq)
    #pl.display((im2-bestmodel)*mask,vmin=-5,vmax=5)

    #best pars:  [ 4.62035985 19.97713692 19.99999571  0.18208773  2.99696345  5.5999840  0.5006771 ]
    #chisq =  1.7596362969096417


    import pdb; pdb.set_trace()

    return pars,bestmodel,chisq


def model_star():
    """ Model a NIRSPEC MSATA image with a star and background."""
    
    pass
