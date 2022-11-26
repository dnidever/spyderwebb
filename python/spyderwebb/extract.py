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
from doppler.spec1d import Spec1D
from . import utils

def profilefit(x,y):
    """ Fit a spectral profile."""
    flux = np.sum(np.maximum(y,0))
    xmean = np.sum(x*np.maximum(y,0))/flux
    xsig = np.sqrt(np.sum((x-xmean)**2 * np.maximum(y,0)/flux))
    # Fit binned Gaussian
    p0 = [np.max(y),xmean,xsig,0.0]
    bnds = [np.array([p0[0]*0.5,xmean-1,0.5*xsig,-0.3*p0[0]]),
            np.array([p0[0]*2,xmean+1,2*xsig,0.3*p0[0]])]
    pars,cov = dln.gaussfit(x,y,initpar=p0,bounds=bnds,binned=True)
    perror = np.sqrt(np.diag(cov))
    return pars,perror
    

def tracing(im,ymid=None,step=25,nbin=50):
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
        profile[~np.isfinite(profile)] = 0.0
        flux = np.nansum(np.maximum(profile,0))
        if flux <= 0:
            continue
        ylo = int(np.floor(lasty-3.0*lastsig))
        yhi = int(np.ceil(lasty+3.0*lastsig))
        slc = slice(ylo,yhi+1)
        profileclip = profile[slc]
        profileclip /= np.sum(np.maximum(profileclip,0))  # normalize
        yclip = y[slc]
        if np.sum(~np.isfinite(profileclip))>0:
            continue
        pars,perror = profilefit(yclip,profileclip)
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
        profile[~np.isfinite(profile)] = 0.0
        flux = np.nansum(np.maximum(profile,0))        
        if flux <= 0:
            continue
        ind = np.argmax(profile)
        ylo = int(np.floor(lasty-3.0*lastsig))
        yhi = int(np.ceil(lasty+3.0*lastsig))
        slc = slice(ylo,yhi+1)
        profileclip = profile[slc]
        profileclip /= np.sum(np.maximum(profileclip,0))  # normalize
        yclip = y[slc]
        if np.sum(~np.isfinite(profileclip))>0:
            continue        
        pars,perror = profilefit(yclip,profileclip)
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
        psf[i,:] = dln.medfilt(psf1[i,:],smlen)
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
    diff = (sim-flux*psf)/serr**2
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
        # Need at least ONE good profile points to measure a flux
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
        # Need at least ONE good profile points to measure a flux
        ngood = np.sum((psf>0.01)*np.isfinite(im),axis=0)
        badcol = (ngood==0)
        flux[badcol] = 0.0
        fluxerr[badcol] = 1e30
        
        return flux,fluxerr


def extract_slit(input_model,slit,verbose=False):
    """ Extract one slit."""

    print('source_name:',slit.source_name)
    print('source_id:',slit.source_id)
    print('slitlet_id:',slit.slitlet_id)
    print('source ra/dec:',slit.source_ra,slit.source_dec)
        
    # Get the data
    im = slit.data.copy()
    err = slit.err.copy()
    wave = slit.wavelength
    ny,nx = im.shape
    bad = (slit.err<=0)
    im[bad] = np.nan
    err[bad] = 1e30
    # Number of good pixels per column
    ngood = np.sum(~bad,axis=0)
    
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
        ap = jextract.get_aperture(im.shape, extract_model.wcs, extract_params)
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
        extract_model.add_position_correction(im.shape)
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

    # I think the trace is at
    # ny/2+offset at Y=1024
    yind = (ny-1)/2+offset
    
    # Get the trace
    x = np.arange(nx)    
    ttab = tracing(im,yind)
    tcoef = robust.polyfit(ttab['x'],ttab['y'],2)
    ytrace = np.polyval(tcoef,x)
    tsigcoef = robust.polyfit(ttab['x'],ttab['ysig'],1)
    ysig = np.polyval(tsigcoef,x)
        
    # Create the mask
    ybin = 3
    yy = np.arange(ny).reshape(-1,1) + np.zeros(nx).reshape(1,-1)
    mask = ((yy >= (ytrace-ybin)) & (yy <= (ytrace+ybin)))

    # Boxcar extraction
    boxflux = np.nansum(mask*im,axis=0)
    
    # Build Gaussian PSF model
    y = np.arange(ny)
    yy = y.reshape(-1,1) + np.zeros(nx).reshape(1,-1)
    gpsf = utils.gauss2dbin(yy,np.ones(nx),ytrace,ysig)
    gpsf /= np.sum(gpsf,axis=0)  # normalize
    # Gaussian PSF extraction
    gflux,gfluxerr,sky,skyerr = extract_psf(im*mask,gpsf,err,skyfit=True)

    # Number of good pixels per column with good PSF
    ngood = np.sum((gpsf>0.01)*np.isfinite(im),axis=0)
    
    # Optimal extraction
    oflux,ofluxerr,otrace,opsf = extract_optimal(im*mask,ytrace,imerr=err)

    # GAUSSIAN extraction looks better!
    flux = gflux
    fluxerr = gfluxerr
    
    # Get the wavelengths
    pmask = (gpsf > 0.01)
    wav = np.nansum(wave*pmask,axis=0)/np.sum(pmask,axis=0) * 1e4  # convert to Angstroms
        
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
    sp = Spec1D(flux,err=fluxerr,wave=wav,instrument='NIRSpec')
    sp.ytrace = ytrace
    sp.source_name = slit.source_name
    sp.source_id = slit.source_id
    sp.slitlet_id = slit.slitlet_id
    sp.source_ra = slit.source_ra
    sp.source_dec = slit.source_dec
    sp.xstart = slit.xstart
    sp.xsize = slit.xsize
    sp.ystart = slit.ystart
    sp.yize = slit.ysize
    sp.offset = offset
    sp.tcoef = tcoef
    sp.tsigcoef = tsigcoef
    #spec = Table((newwav,flux,fluxerr,trace),names=['wave','flux','flux_error','ytrace'])

    ## Save the file
    #filename = slit.source_name+'_'+filebase+'.fits'
    #print('Writing spectrum to ',filename)
    #sp.write(filename,overwrite=True)

    return sp
