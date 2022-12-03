# Packages that allow us to get information about objects:
import asdf
import os
import numpy as np
from astropy.table import Table
from dlnpyutils import utils as dln,robust

from jwst.extract_1d.extract_1d_step import Extract1dStep
from jwst.extract_1d import extract as jextract

# Astropy tools:
from astropy.io import fits
from astropy.time import Time
from doppler.spec1d import Spec1D
from scipy.ndimage import median_filter,generic_filter
from . import utils

import matplotlib
import matplotlib.pyplot as plt
from dlnpyutils import plotting as pl


# Ignore these warnings
import warnings
warnings.filterwarnings("ignore", message="OptimizeWarning: Covariance of the parameters could not be estimated")

def nanmedfilt(x,size,mode='reflect'):
    return generic_filter(x, np.nanmedian, size=size)

def profilefit(x,y):
    """ Fit a spectral profile."""
    flux = np.sum(np.maximum(y,0))
    xmean = np.sum(x*np.maximum(y,0))/flux
    xsig = np.sqrt(np.sum((x-xmean)**2 * np.maximum(y,0)/flux))
    xsig = np.maximum(xsig,0.1)
    # Fit binned Gaussian
    p0 = [np.max(y),xmean,xsig,0.0]
    bnds = [np.array([p0[0]*0.5,xmean-1,0.5*xsig,-0.3*p0[0]]),
            np.array([p0[0]*2,xmean+1,2*xsig,0.3*p0[0]])]
    if np.sum((bnds[0][:] >= bnds[1][:]))>0:
        print('problem in profilefit')
        import pdb; pdb.set_trace()

    try:
        pars,cov = dln.gaussfit(x,y,initpar=p0,bounds=bnds,binned=True)
        perror = np.sqrt(np.diag(cov))
        return pars,perror        
    except:
        return None,None
    

def tracing(im,err,ymid=None,step=15,nbin=25):
    """ Trace a spectrum. Assumed to be in the horizontal direction."""
    ny,nx = im.shape
    y,x = np.arange(ny),np.arange(nx)
    
    # Find ymid if not input
    if ymid is None:
        #tot = np.nanmedian(np.maximum(im[:,nx//2-50:nx//2+50],0),axis=1)
        tot = np.nanmedian(np.maximum(im,0),axis=1)        
        ymid = np.argmax(tot)

    # Trace using binned profiles, starting at middle
    nsteps = (nx//2)//step
    #if nsteps % 2 == 0: nsteps-=1
    lasty = ymid
    lastsig = 1.0
    xmnarr = []
    yhtarr = []    
    ymidarr = []
    ysigarr = []
    # Forwards
    for i in range(nsteps):
        xmn = nx//2 + i*step
        xlo = xmn - nbin//2
        xhi = xmn + nbin//2 + 1
        profile = np.nanmedian(im[:,xlo:xhi],axis=1)
        profileerr = np.nanmedian(err[:,xlo:xhi],axis=1)        
        profile[~np.isfinite(profile)] = 0.0
        profileerr[~np.isfinite(profileerr) | (profileerr<=0)] = 1e30
        flux = np.nansum(np.maximum(profile,0))
        snr = np.nanmax(profile/profileerr)  # max S/N
        if flux <= 0 or snr<5:
            continue
        ylo = np.maximum(int(np.floor(lasty-3.0*lastsig)),0)
        yhi = np.minimum(int(np.ceil(lasty+3.0*lastsig)),ny)
        slc = slice(ylo,yhi+1)
        profileclip = profile[slc]
        profileclip /= np.sum(np.maximum(profileclip,0))  # normalize
        yclip = y[slc]
        if np.sum(~np.isfinite(profileclip))>0:
            continue
        if len(yclip)==0:
            print('no pixels')
            import pdb; pdb.set_trace()
        pars,perror = profilefit(yclip,profileclip)
        if pars is None:
            continue
        xmnarr.append(xmn)
        yhtarr.append(pars[0])        
        ymidarr.append(pars[1])
        ysigarr.append(pars[2])
        # Remember
        lasty = pars[1]
        lastsig = pars[2]
        
    # Backwards
    lasty = ymid
    lastsig = 0.5
    for i in np.arange(1,nsteps):
        xmn = nx//2 - i*step
        xlo = xmn - nbin//2
        xhi = xmn + nbin//2 + 1
        profile = np.nanmedian(im[:,xlo:xhi],axis=1)
        profileerr = np.nanmedian(err[:,xlo:xhi],axis=1)        
        profile[~np.isfinite(profile)] = 0.0
        profileerr[~np.isfinite(profileerr) | (profileerr<=0)] = 1e30
        flux = np.nansum(np.maximum(profile,0))
        snr = np.nanmax(profile/profileerr)  # max S/N
        if flux <= 0 or snr<5:
            continue
        ind = np.argmax(profile)
        ylo = np.maximum(int(np.floor(lasty-3.0*lastsig)),0)
        yhi = np.minimum(int(np.ceil(lasty+3.0*lastsig)),ny)
        slc = slice(ylo,yhi+1)
        profileclip = profile[slc]
        profileclip /= np.sum(np.maximum(profileclip,0))  # normalize
        yclip = y[slc]
        if np.sum(~np.isfinite(profileclip))>0:
            continue        
        if len(yclip)==0:
            print('no pixels')
            import pdb; pdb.set_trace()
        pars,perror = profilefit(yclip,profileclip)
        if pars is None:
            continue        
        xmnarr.append(xmn)
        yhtarr.append(pars[0])        
        ymidarr.append(pars[1])
        ysigarr.append(pars[2])
        # Remember
        lasty = pars[1]
        lastsig = pars[2]

    ttab = Table((xmnarr,ymidarr,ysigarr,yhtarr),names=['x','y','ysig','amp'])
    ttab.sort('x')
        
    return ttab

def optimalpsf(im,ytrace,err=None,off=10,backoff=50,smlen=31):
    """ Compute the PSF from the image using "optimal extraction" techniques."""
    ny,nx = im.shape
    yest = np.nanmedian(ytrace)
    # Get the subimage
    yblo = int(np.maximum(yest-backoff,0))
    ybhi = int(np.minimum(yest+backoff,ny))
    nback = ybhi-yblo
    # Background subtract
    med = np.nanmedian(im[yblo:ybhi,:],axis=0)
    medim = np.zeros(nback).reshape(-1,1) + med.reshape(1,-1)
    subim = im[yblo:ybhi,:]-medim
    suberr = imerr[yblo:ybhi,:]
    # Make sure the arrays are float64
    subim = subim.astype(float)
    suberr = suberr.astype(float)    
    # Mask other parts of the image
    ylo = ytrace-off - yblo
    yhi = ytrace+off - yblo
    yy = np.arange(nback).reshape(-1,1)+np.zeros(nx)
    mask = (yy >= ylo) & (yy <= yhi)
    sim = subim*mask
    serr = suberr*mask
    badpix = (serr <= 0)
    serr[badpix] = 1e20
    # Compute the profile/probability matrix from the image
    tot = np.nansum(np.maximum(sim,0),axis=0)
    tot[(tot<=0) | ~np.isfinite(tot)] = 1
    psf1 = np.maximum(sim,0)/tot
    psf = np.zeros(psf1.shape,float)
    for i in range(nback):
        psf[i,:] = dln.medfilt(psf1[i,:],smlen)
        #psf[i,:] = utils.gsmooth(psf1[i,:],smlen)        
    psf[(psf<0) | ~np.isfinite(psf)] = 0
    totpsf = np.nansum(psf,axis=0)
    totpsf[(totpsf<=0) | (~np.isfinite(totpsf))] = 1
    psf /= totpsf
    psf[(psf<0) | ~np.isfinite(psf)] = 0

    return psf
    
    
def extract_optimal(im,ytrace,imerr=None,verbose=False,off=10,backoff=50,smlen=31):
    """ Extract a spectrum using optimal extraction (Horne 1986)"""
    ny,nx = im.shape
    yest = np.nanmedian(ytrace)
    # Get the subo,age
    yblo = int(np.maximum(yest-backoff,0))
    ybhi = int(np.minimum(yest+backoff,ny))
    nback = ybhi-yblo
    # Background subtract
    med = np.nanmedian(im[yblo:ybhi,:],axis=0)
    medim = np.zeros(nback).reshape(-1,1) + med.reshape(1,-1)
    subim = im[yblo:ybhi,:]-medim
    suberr = imerr[yblo:ybhi,:]
    # Make sure the arrays are float64
    subim = subim.astype(float)
    suberr = suberr.astype(float)    
    # Mask other parts of the image
    ylo = ytrace-off - yblo
    yhi = ytrace+off - yblo
    yy = np.arange(nback).reshape(-1,1)+np.zeros(nx)
    mask = (yy >= ylo) & (yy <= yhi)
    sim = subim*mask
    serr = suberr*mask
    badpix = (serr <= 0)
    serr[badpix] = 1e20
    # Compute the profile/probability matrix from the image
    tot = np.nansum(np.maximum(sim,0),axis=0)
    tot[(tot<=0) | ~np.isfinite(tot)] = 1
    psf1 = np.maximum(sim,0)/tot
    psf = np.zeros(psf1.shape,float)
    for i in range(nback):
        psf[i,:] = nanmedfilt(psf1[i,:],smlen)
        #psf[i,:] = dln.medfilt(psf1[i,:],smlen)        
        #psf[i,:] = utils.gsmooth(psf1[i,:],smlen)        
    psf[(psf<0) | ~np.isfinite(psf)] = 0
    totpsf = np.nansum(psf,axis=0)
    totpsf[(totpsf<=0) | (~np.isfinite(totpsf))] = 1
    psf /= totpsf
    psf[(psf<0) | ~np.isfinite(psf)] = 0
    # Compute the weights
    wt = psf**2/serr**2
    wt[(wt<0) | ~np.isfinite(wt)] = 0
    totwt = np.nansum(wt,axis=0)
    badcol = (totwt<=0)
    totwt[badcol] = 1
    # Compute the flux and flux error
    flux = np.nansum(psf*sim/serr**2,axis=0)/totwt
    fluxerr = np.sqrt(1/totwt)    
    fluxerr[badcol] = 1e30  # bad columns
    # Recompute the trace
    trace = np.nansum(psf*yy,axis=0)+yblo
    
    # Check for outliers
    diff = (sim-flux*psf)/serr
    bad = (diff > 25)
    if np.nansum(bad)>0:
        # Mask bad pixels
        sim[bad] = 0
        serr[bad] = 1e20
        # Recompute the flux
        wt = psf**2/serr**2
        totwt = np.nansum(wt,axis=0)
        badcol = (totwt<=0)
        totwt[badcol] = 1        
        flux = np.nansum(psf*sim/serr**2,axis=0)/totwt
        fluxerr = np.sqrt(1/totwt)
        fluxerr[badcol] = 1e30  # bad columns
        # Recompute the trace
        trace = np.nansum(psf*yy,axis=0)+yblo

    # Need at least ONE good profile point to measure a flux
    ngood = np.sum((psf>0.01)*np.isfinite(im),axis=0)
    badcol = (ngood==0)
    flux[badcol] = 0.0
    fluxerr[badcol] = 1e30 
        
    return flux,fluxerr,trace,psf


def extract_psf(im,psf,err=None,skyfit=True):
    """ Extract spectrum with a PSF."""

    if err is None:
        err = np.ones(im.shape,float)
    # Fit the sky
    if skyfit:
        # Compute the weights
        # If you are solving for flux and sky, then
        # you need to do 1/err**2 weighting
        wt = 1/err**2
        wt[(wt<0) | ~np.isfinite(wt)] = 0
        totwt = np.sum(wt,axis=0)
        badcol = (totwt<=0)
        totwt[badcol] = 1
        # Perform weighted linear regression
        flux,fluxerr,sky,skyerr = utils.weightedregression(psf,im,wt,zero=False)
        # Compute the flux and flux error
        fluxerr[badcol] = 1e30  # bad columns
        # Need at least ONE good profile point to measure a flux
        ngood = np.sum((psf>0.01)*np.isfinite(im),axis=0)
        badcol = (ngood==0)
        flux[badcol] = 0.0
        fluxerr[badcol] = 1e30
        
        return flux,fluxerr,sky,skyerr        
        
    # Only solve for flux
    #  assume sky was already subtracted
    else:
        wt = psf**2/err**2
        totwt = np.sum(wt,axis=0)
        badcol = (totwt<=0)
        totwt[badcol] = 1
        # Perform weighted linear regression
        flux,fluxerr = utils.weightedregression(psf,im,wt,zero=True)
        # Compute the flux and flux error
        fluxerr[badcol] = 1e30  # bad columns
        # Need at least ONE good profile point to measure a flux
        ngood = np.sum((psf>0.01)*np.isfinite(im),axis=0)
        badcol = (ngood==0)
        flux[badcol] = 0.0
        fluxerr[badcol] = 1e30
        
        return flux,fluxerr

def extractcol(im,err,psf):
    # Optimal extraction of a single column
    wt = psf**2/err**2
    wt[(wt<0) | ~np.isfinite(wt)] = 0
    totwt = np.nansum(wt,axis=0)
    if totwt <= 0: totwt=1
    # Compute the flux and flux error
    flux = np.nansum(psf*im/err**2,axis=0)/totwt
    fluxerr = np.sqrt(1/totwt)
    if np.isfinite(flux)==False:
        fluxerr = 1e30  # bad columns
    return flux,fluxerr

def findworstcolpix(im,err,psf):
    """ Find worst outlier pixel in a column"""
    n = len(im)
    gd = np.arange(n)    
    # Loop over the good pixels and take on away each time
    #  then recompute the flux and rchisq
    rchiarr = np.zeros(n)
    fluxarr = np.zeros(n)
    fluxerrarr = np.zeros(n)
    for j in range(n):
        ind = gd.copy()
        ind = np.delete(ind,j)
        tflx,tflxerr = extractcol(im[ind],err[ind],psf[ind])
        fluxarr[j] = tflx
        fluxerrarr[j] = tflxerr
        rchi1 = np.sum((im[ind]-tflx*psf[ind])**2/err[ind]**2)/(n-1)
        rchiarr[j] = rchi1
    bestind = np.argmin(rchiarr)
    bestrchi = rchiarr[bestind]
    bestflux = fluxarr[bestind]
    bestfluxerr = fluxerrarr[bestind]
    return bestind,bestrchi,bestflux,bestfluxerr

def fixbadpixels(im,err,psf):
    """ Fix outlier pixels using the PSF."""

    ny,nx = im.shape
    # Compute chi-squared for each column and check outliers for bad pixels
    mask = (psf > 0.01)
    # Recompute the flux
    wt = psf**2/err**2
    totwt = np.nansum(wt,axis=0)
    badcol = (totwt<=0)
    totwt[badcol] = 1        
    flux = np.nansum(psf*im/err**2,axis=0)/totwt
    # Calculate model and chisq
    model = psf*flux.reshape(1,-1)
    chisq = np.sum((model-im*mask)**2/err**2,axis=0)
    ngood = np.sum(np.isfinite(mask*im)*mask,axis=0)
    rchisq = chisq/np.maximum(ngood,1)
    #smlen = np.minimum(7,nx)
    #medrchisq = nanmedfilt(rchisq,smlen,mode='mirror')
    #sigrchisq = dln.mad(rchisq-medrchisq)
    medrchisq = np.maximum(np.nanmedian(rchisq),1.0)
    sigrchisq = dln.mad(rchisq-medrchisq,zero=True)
    coltofix, = np.where(((rchisq-medrchisq > 5*sigrchisq) | ~np.isfinite(rchisq)) & (rchisq>5) & (ngood>0))

    fixmask = np.zeros(im.shape,bool)
    fixim = im.copy()
    fixflux = flux.copy()
    fixfluxerr = flux.copy()    
    
    # Loop over columns to try to fix
    for i,c in enumerate(coltofix):
        cim = im[:,c]
        cerr = err[:,c]
        cpsf = psf[:,c]
        cflux = flux[c]
        gd, = np.where(mask[:,c]==True)
        ngd = len(gd)
        rchi = np.sum((cim[gd]-cflux*cpsf[gd])**2/cerr[gd]**2)/ngd

        # We need to try each pixel separately because if there is an
        # outlier pixel then the flux will be bad and all pixels will be "off"
        
        # While loop to fix bad pixels
        prevrchi = rchi
        count = 0
        fixed = True
        while ((fixed==True) & (ngd>2)):
            # Find worse outlier pixel in a column
            bestind,bestrchi,bestflux,bestfluxerr = findworstcolpix(cim[gd],cerr[gd],cpsf[gd])
            # Make sure it is a decent improvement
            fixed = False
            if bestrchi<0.8*prevrchi and prevrchi>5:
                curfixind = gd[bestind]
                fixmask[curfixind,c] = True
                #print('Fixing pixel ['+str(curfixind)+','+str(c)+'] ',prevrchi,bestrchi)
                # Find current and previous fixed pixels
                #  need to replace all of their values
                #  using this new flux
                fixind, = np.where(fixmask[:,c]==True)
                fixim[fixind,c] = cpsf[fixind]*bestflux
                fixflux[c] = bestflux
                fixfluxerr[c] = bestfluxerr
                gd = np.delete(gd,bestind)  # delete the pixel from the good pixel list
                ngd -= 1
                fixed = True
                prevrchi = bestrchi
            count += 1            
            
    return fixim,fixmask,fixflux,fixfluxerr


def extract_slit(input_model,slit,ratehdu,bratehdu=None,verbose=False,plotbase='extract'):
    """ Extract one slit."""

    print('source_name:',slit.source_name)
    print('source_id:',slit.source_id)
    print('slitlet_id:',slit.slitlet_id)
    print('source ra/dec:',slit.source_ra,slit.source_dec)
    print('ny/nx:',*slit.data.shape)

    # Clip the X ends
    #  sometimes there are fully masked columns at the ends
    goodpix, = np.where((np.sum(np.isfinite(slit.data),axis=0)>0) & (np.sum(slit.err>=0,axis=0)>0))
    if len(goodpix)==0:
        print('No good pixels')
        return None
    xlo = goodpix[0]
    xhi = goodpix[-1]
    xstart = slit.xstart+xlo
    xsize = xhi-xlo+1
    if xlo>0:
        print('Trimming first '+str(xlo)+' fully masked columns')
    if xhi < (slit.xsize-1):
        print('Trimming last '+str((slit.xsize-1)-xhi)+' fully masked columns')
    
    # Get the data
    slim = slit.data.copy()[:,xlo:xhi+1]
    slerr = slit.err.copy()[:,xlo:xhi+1]
    slwave = slit.wavelength[:,xlo:xhi+1]
    ny,nx = slim.shape
    slbad = (slerr<=0)
    slim[slbad] = np.nan
    slerr[slbad] = 1e30
    # Number of good pixels per column
    nslgood = np.sum(~slbad,axis=0)
    
    # Extend the wavelength information across the full image
    #  normally it only covers one section
    wave = slwave.copy()
    y = np.arange(ny)
    ngoodwave = np.sum(np.isfinite(wave),axis=0)
    wcoef = np.zeros([nx,2],float)
    for i in range(nx):
        if ngoodwave[i]>2:
            w = wave[:,i]            
            good = np.isfinite(w)
            bad = ~np.isfinite(w)
            wcoef1 = np.polyfit(y[good],w[good],1)
            wcoef[i,:] = wcoef1
            wave[bad,i] = np.polyval(wcoef1,y[bad])


    # NEED to mask bad pixels in rate images better!!!!!
    # Use the DQ information!!!
            
    # Get the "raw" rate data
    im = ratehdu[1].data
    err = ratehdu[2].data
    bad = (err<=0)
    im[bad] = np.nan
    err[bad] = 1e30
    # Get the "raw" rate background data
    if bratehdu is not None:
        bim = bratehdu[1].data
        berr = bratehdu[2].data    
        bbad = (berr<=0)
        bim[bbad] = np.nan
        berr[bbad] = 1e30
        
    if np.sum(nslgood)==0:
        print('No data to extract')
        return None
    
    ## Get the reference file
    if True:
        step = Extract1dStep()
        extract_ref = step.get_reference_file(input_model,'extract1d')
        extract_ref_dict = jextract.open_extract1d_ref(extract_ref, input_model.meta.exposure.type)
    
        # Get extraction parameters, from extract.create_extraction()
        slitname = slit.name
        sp_order = jextract.get_spectral_order(slit)        
        meta_source = slit
        smoothing_length = None
        use_source_posn = True
        extract_params = jextract.get_extract_parameters(extract_ref_dict,meta_source,slitname,
                                                         sp_order,input_model.meta,step.smoothing_length,
                                                         step.bkg_fit,step.bkg_order,use_source_posn)
        extract_params['dispaxis'] = extract_ref_dict['apertures'][0]['dispaxis']
    
        # Get extraction model, extract.extract_one_slit()
        # If there is an extract1d reference file (there doesn't have to be), it's in JSON format.
        extract_model = jextract.ExtractModel(input_model=input_model, slit=slit, **extract_params)
        ap = jextract.get_aperture(slim.shape, extract_model.wcs, extract_params)
        extract_model.update_extraction_limits(ap)
        
        if extract_model.use_source_posn:
            #if prev_offset == OFFSET_NOT_ASSIGNED_YET:  # Only call this method for the first integration.
            offset, locn = extract_model.offset_from_offset(input_model, slit)
    
            #if offset is not None and locn is not None:
            #    log.debug(f"Computed source offset={offset:.2f}, source location={locn:.2f}")
    
            if not extract_model.use_source_posn:
                offset = 0.
            #else:
            #    offset = prev_offset
        else:
            offset = 0.
    
        extract_model.position_correction = offset
    
        # Add the source position offset to the polynomial coefficients, or shift the reference image
        # (depending on the type of reference file).
        extract_model.add_position_correction(slim.shape)
        extract_model.log_extraction_parameters()
        extract_model.assign_polynomial_limits()
    
        # get disp_range, from extract.ExtractModel.extract()
        if extract_model.dispaxis == jextract.HORIZONTAL:
            slice0 = int(round(extract_model.xstart))
            slice1 = int(round(extract_model.xstop)) + 1
            #x_array = np.arange(slice0, slice1, dtype=np.float64)
            #y_array = np.empty(x_array.shape, dtype=np.float64)
            #y_array.fill((extract_model.ystart + extract_model.ystop) / 2.)
        else:
            slice0 = int(round(extract_model.ystart))
            slice1 = int(round(extract_model.ystop)) + 1
            #y_array = np.arange(slice0, slice1, dtype=np.float64)
            #x_array = np.empty(y_array.shape, dtype=np.float64)
            #x_array.fill((extract_model.xstart + extract_model.xstop) / 2.)
        disp_range = [slice0, slice1]  # Range (slice) of pixel numbers in the dispersion direction.
          
        # Get the trace, from extract1d.extract1d()
        p_src = extract_model.p_src
        srclim = []                 # this will be a list of lists, like p_src
        n_srclim = len(p_src)
    
        for i in range(n_srclim):
            lower = p_src[i][0]
            upper = p_src[i][1]
            if extract_model.independent_var.startswith("wavelength"):    # OK if 'wavelengths'
                srclim.append([lower(lambdas), upper(lambdas)])
            else:
                # Temporary array for the independent variable.
                pixels = np.arange(disp_range[0], disp_range[1], dtype=np.float64)
                srclim.append([lower(pixels), upper(pixels)])

    ## Polynomial coefficients for traces
    ## y(1024) = 0
    #coef0 = np.array([ 4.59440169e-06, -1.86120908e-04, -4.83294422e+00])
    #
    ## Get peak at X=1024
    #tot = np.nansum(np.maximum(im,0),axis=1)
    #yind = np.argmax(tot)
    ## Trace for this star
    #coef = coef0.copy()
    #coef[2] += yind
    #x = np.arange(nx)
    #ytrace = np.polyval(coef,x)

    print('offset: ',offset)
    
    # Extract the CAL 2D SLIT IMAGE
    #------------------------------

    # I think the trace is at
    # ny/2+offset at Y=1024
    yind = (ny-1)/2+offset
    
    # Get the trace
    x = np.arange(nx)
    slttab = tracing(slim,slerr,yind)

    if len(slttab)==0:
        print('Problem - no trace found')
        return None

    try:
        if len(slttab)<3:
            sltcoef = np.array([np.median(slttab['y'])])
            sltsigcoef = np.array([np.median(slttab['ysig'])])            
        else:
            sltcoef = robust.polyfit(slttab['x'],slttab['y'],2)
            sltsigcoef = robust.polyfit(slttab['x'],slttab['ysig'],1)            
        slytrace = np.polyval(sltcoef,x)
        slysig = np.polyval(sltsigcoef,x)
    except:
        print('tracing coefficient problem')
        import pdb; pdb.set_trace()

        
    # Create the mask
    ybin = 3
    yy = np.arange(ny).reshape(-1,1) + np.zeros(nx).reshape(1,-1)
    slmask = ((yy >= (slytrace-ybin)) & (yy <= (slytrace+ybin)))

    # Boxcar extraction
    slboxflux = np.nansum(slmask*slim,axis=0)
    
    # Build Gaussian PSF model
    y = np.arange(ny)
    yy = y.reshape(-1,1) + np.zeros(nx).reshape(1,-1)
    slgpsf = utils.gauss2dbin(yy,np.ones(nx),slytrace,slysig)
    slgpsf /= np.sum(slgpsf,axis=0)  # normalize
    # Gaussian PSF extraction
    slgflux,slgfluxerr,slsky,slskyerr = extract_psf(slim*slmask,slgpsf,slerr,skyfit=True)

    # Number of good pixels per column with good PSF
    slngood = np.sum((slgpsf>0.01)*np.isfinite(slim),axis=0)
    
    # Optimal extraction
    sloflux,slofluxerr,slotrace,slopsf = extract_optimal(slim*slmask,slytrace,imerr=slerr)

    # GAUSSIAN extraction looks better!
    #flux = gflux
    #fluxerr = gfluxerr
    
    # Get the wavelengths
    slpmask = (slgpsf > 0.01)
    #wav = np.nansum(wave*pmask,axis=0)/np.sum(pmask,axis=0) * 1e4  # convert to Angstroms
    #wav = np.nansum(slwave*slgpsf,axis=0)/np.sum(slgpsf*np.isfinite(slwave*slgpsf),axis=0) * 1e4  # convert to Angstroms

    # Save some diagnostic plots    
    matplotlib.use('Agg')
    fig = plt.figure(figsize=(12,7))
    medflux = np.nanmedian(sloflux)
    vmin = -medflux/3.
    vmax = medflux
    pl.display(slim,xtitle='X',ytitle='Y',vmin=vmin,vmax=vmax,title='SLIT CAL DATA - SLIT_ID '+str(slit.source_id))
    pl.oplot(slytrace,c='red')
    plt.savefig(plotbase+'_slitdata.png',bbox_inches='tight')
    pl.display(slopsf,xtitle='X',ytitle='Y',vmin=0,vmax=1.2*np.max(slopsf),title='SLIT CAL Optimal PSF - SLIT_ID '+str(slit.source_id))
    pl.oplot(slytrace,c='red')
    plt.savefig(plotbase+'_slitopsf.png',bbox_inches='tight')
    matplotlib.use('MacOSX')
    
    # Extract the RATE IMAGE with background subtraction
    #---------------------------------------------------
    
    # Now use the rate images to do the extraction
    # the XSTART/YSTART are in 1-based indexing
    slc = (slice(slit.ystart-1,slit.ystart+slit.ysize-1),slice(xstart-1,xstart+slit.xsize-1))
    sim = im[slc]
    serr = err[slc]
    sbim = bim[slc]
    sberr = berr[slc]
    # Measure the scaling factor
    goodpix = ((sim/serr>5) & (slim/slerr>5) & slpmask)
    if goodpix.sum()==0:
        goodpix = ((sim/serr>3) & (slim/slerr>3) & slpmask)
    if goodpix.sum()==0:
        print('No good pixels in common to CAL and RATE image')
        return None
    scale = np.nanmedian(slim[goodpix]/sim[goodpix])
    print('scale = ',scale)
    sim *= scale
    serr *= scale
    sbim *= scale
    sberr *= scale
    ytrace0 = slytrace.copy()
    mask0 = slpmask.copy()
    ystart = slit.ystart
    ysize = slit.ysize
    
    # Trace is close to the edge, use larger range
    if ((ny-1)-np.max(slytrace) < 4) or (np.min(slytrace)<4):
        if ((ny-1)-np.max(slytrace) < 4):
            nextend = int(np.ceil(4-((ny-1)-np.max(slytrace))))
            ystart = slit.ystart
            ysize = slit.ysize+nextend
            print('Extending the subimage '+str(nextend)+' pixels at the TOP')
            # Extend the wavelength array
            wave = np.vstack((wave,np.zeros([nextend,nx])+np.nan))
            for i in range(nx):
                wave[ny:ny+nextend,i] = np.polyval(wcoef[i,:],np.arange(nextend)+ny)
            # Extend the initial mask
            mask0 = np.vstack((mask0,np.zeros([nextend,nx],bool)))
        else:
            nextend = int(np.floor(4-np.min(slytrace)))
            ystart = slit.ystart-nextend
            ysize = slit.ysize+nextend            
            ytrace0 += nextend
            print('Extending the subimage '+str(nextend)+' pixels at the BOTTOM')
            # Extend the wavelength array
            wave = np.vstack((np.zeros([nextend,nx])+np.nan,wave))
            for i in range(nx):
                wave[0:nextend,i] = np.polyval(wcoef[i,:],np.arange(nextend)-nextend)
            # Extend the initial mask
            mask0 = np.vstack((np.zeros([nextend,nx],bool),mask0))
        ny += nextend
        y = np.arange(ny)
        yy = y.reshape(-1,1) + np.zeros(nx).reshape(1,-1)        
        slc = (slice(ystart-1,ystart+ysize-1),slice(xstart-1,xstart+nx-1))
        sim = im[slc]
        serr = err[slc]
        sbim = bim[slc]
        sberr = berr[slc]
        sim *= scale
        serr *= scale
        sbim *= scale
        sberr *= scale  
    
    # Redo the tracing
    tmask = dln.convolve(mask0.astype(float),np.ones((3,3)))
    tmask[tmask>0] /= tmask[tmask>0]   # normalize to 0/1
    tmask = tmask.astype(bool)
    ttab = tracing(sim*tmask,serr,np.median(ytrace0))

    if len(ttab)==0:
        print('No trace points')
        return None
    
    try:
        if len(ttab)<3:
            tcoef = np.array([np.median(ttab['y'])])
            tsigcoef = np.array([np.median(ttab['ysig'])])            
        else:
            tcoef = robust.polyfit(ttab['x'],ttab['y'],2)
            tsigcoef = robust.polyfit(ttab['x'],ttab['ysig'],1)            
        ytrace = np.polyval(tcoef,x)
        ysig = np.polyval(tsigcoef,x)
    except:
        print('tracing coefficient problem')
        import pdb; pdb.set_trace()

    # Create the mask
    ybin = 3
    mask = ((yy >= (ytrace-ybin)) & (yy <= (ytrace+ybin)))

    # Get the background from the background image
    if bratehdu is not None:
        temp1 = sbim.copy()
        temp1[~mask] = np.nan
        sky1 = np.nanmedian(temp1,axis=0)
        diff1 = sbim.copy()-sky1.reshape(1,-1)
        sig1 = dln.mad(diff1[mask])
        temp2 = sbim.copy()
        temp2[(~mask) | (np.abs(diff1)>3*sig1)] = np.nan
        sky2 = np.nanmean(temp2,axis=0)
        sky = np.zeros(ny).reshape(-1,1) + sky2.reshape(1,-1)
    else:
        sky = np.zeros((ny,nx))
    sim -= sky   # subtract background
        
    # Build Gaussian PSF model
    y = np.arange(ny)
    yy = y.reshape(-1,1) + np.zeros(nx).reshape(1,-1)
    gpsf = utils.gauss2dbin(yy,np.ones(nx),ytrace,ysig)
    gpsf /= np.sum(gpsf,axis=0)  # normalize
    # Gaussian PSF extraction
    gflux,gfluxerr,sky,skyerr = extract_psf(sim*mask,gpsf,serr,skyfit=True)

    # Number of good pixels per column with good PSF
    ngood = np.sum((gpsf>0.01)*np.isfinite(sim),axis=0)

    # Optimal extraction
    oflux1,ofluxerr1,otrace1,opsf1 = extract_optimal(sim*mask,ytrace,imerr=serr)

    # Fix bad pixels using the optimal PSF
    fixim,fixmask,fixflux,fixfluxerr = fixbadpixels(sim,serr,opsf1)

    # REJECT BAD PIXELS and redo the optimal extraction
    oflux,ofluxerr,otrace,opsf = extract_optimal(fixim*mask,ytrace,imerr=serr)

    # Boxcar extraction of the fixed image
    boxflux = np.nansum(mask*fixim,axis=0)

    # Optimal extraction looks a little bit better than the boxcar extraction
    #  lower scatter
    flux = oflux
    fluxerr = ofluxerr

    # Save some diagnostic plots
    matplotlib.use('Agg')
    fig = plt.figure(figsize=(12,7))
    # Rate data
    medflux = np.nanmedian(oflux)
    vmin = -medflux/3.
    vmax = medflux
    pl.display(sim,xtitle='X',ytitle='Y',vmin=vmin,vmax=vmax,title='RATE DATA - SLIT_ID '+str(slit.source_id))
    pl.oplot(ytrace,c='red')
    plt.savefig(plotbase+'_ratedata.png',bbox_inches='tight')
    # Rate optimal PSF
    pl.display(opsf,xtitle='X',ytitle='Y',vmin=0,vmax=0.5,title='RATE Optimal PSF - SLIT_ID '+str(slit.source_id))
    pl.oplot(ytrace,c='red')
    plt.savefig(plotbase+'_rateopsf.png',bbox_inches='tight')
    # Flux
    plt.clf()
    plt.plot(oflux,label='Optimal',linewidth=2)
    plt.plot(gflux,label='Gaussian')
    plt.plot(boxflux,label='Boxcar',linestyle='dashed')
    plt.xlabel('X')
    plt.ylabel('Flux')
    plt.legend()
    plt.savefig(plotbase+'_rateflux.png',bbox_inches='tight')
    matplotlib.use('MacOSX')
    
    # Get the wavelengths
    pmask = (opsf > 0.01)
    #wav = np.nansum(wave*pmask,axis=0)/np.sum(pmask,axis=0) * 1e4  # convert to Angstroms
    wav = np.nansum(wave*opsf,axis=0)/np.sum(opsf*np.isfinite(wave*opsf),axis=0) * 1e4  # convert to Angstroms

    # Should I use boxcar or optimal extraction?
    # I think boxcar is more precise at high S/N (in the absence of outlier pixels)
    # while optimal extraction is better at low-S/N
    
    # Apply slit correction
    srcxpos = slit.source_xpos
    srcypos = slit.source_ypos
    # SLIT correction, srcxpos is source position in slit
    # the slit is 2 pixels wide
    dwave = np.gradient(wav)
    newwav = wav+2*srcxpos*dwave
    print('Applying slit correction: %.2f pixels' % (2*srcxpos))

    # Apply relative flux calibration correction
    
    # Put it all together
    sp = Spec1D(flux,err=fluxerr,wave=wav,mask=(fluxerr>1e20),instrument='NIRSpec')
    sp.date = input_model.meta.date
    sp.jd = Time(input_model.meta.date).jd
    sp.ytrace = ytrace
    sp.source_name = slit.source_name
    sp.source_id = slit.source_id
    sp.slitlet_id = slit.slitlet_id
    sp.source_ra = slit.source_ra
    sp.source_dec = slit.source_dec
    sp.xstart = xstart
    sp.xsize = xsize
    sp.ystart = ystart  # might have been extended
    sp.ysize = ysize
    sp.offset = offset
    sp.tcoef = tcoef
    sp.tsigcoef = tsigcoef
    
    return sp
