# Packages that allow us to get information about objects:
import asdf
import os

# Astropy tools:
from astropy.io import fits

# JWST data models
from jwst import datamodels

# The entire calwebb_spec2 pipeline
from jwst.pipeline.calwebb_spec2 import Spec2Pipeline

# Individual steps that make up calwebb_spec2 and datamodels
from jwst.assign_wcs.assign_wcs_step import AssignWcsStep
from jwst.background.background_step import BackgroundStep
from jwst.imprint.imprint_step import ImprintStep
from jwst.msaflagopen.msaflagopen_step import MSAFlagOpenStep
from jwst.extract_2d.extract_2d_step import Extract2dStep
from jwst.srctype.srctype_step import SourceTypeStep
from jwst.master_background.master_background_step import MasterBackgroundStep
from jwst.wavecorr.wavecorr_step import WavecorrStep
from jwst.flatfield.flat_field_step import FlatFieldStep
from jwst.straylight.straylight_step import StraylightStep
from jwst.fringe.fringe_step import FringeStep
from jwst.pathloss.pathloss_step import PathLossStep
from jwst.barshadow.barshadow_step import BarShadowStep
from jwst.photom.photom_step import PhotomStep
from jwst.resample import ResampleSpecStep
from jwst.cube_build.cube_build_step import CubeBuildStep
from jwst.extract_1d.extract_1d_step import Extract1dStep
from jwst.extract_1d import extract

def reduce(files):
    """ This reduces the JWST NIRSpec MSA data """
    pass

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
    

def tracing(im,ymid=None,step=50,nbin=50):
    """ Trace a spectrum. Assumed to be in the horizontal direction."""
    ny,nx = im.shape
    y,x = np.arange(ny),np.arange(nx)
    
    # Find ymid if not input
    if ymid is None:
        tot = np.nanmedian(np.maximum(im[:,nx//2-50:nx//2+50],0),axis=1)
        ymid = np.argmax(tot)
        
    # Trace using binned profiles, starting at middle
    nsteps = (nx//2)//step
    #if nsteps % 2 == 0: nsteps-=1
    lasty = ymid
    lastsig = 1.0
    xmnarr = []
    ymidarr = []
    ysigarr = []
    # Forwards
    for i in range(nsteps):
        xmn = nx//2 + i*step
        xlo = xmn - nbin//2
        xhi = xmn + nbin//2 + 1
        profile = np.nanmedian(im[:,xlo:xhi],axis=1)
        ind = np.argmax(profile)
        ylo = int(np.floor(lasty-2.5*lastsig))
        yhi = int(np.ceil(lasty+2.5*lastsig))
        slc = slice(ylo,yhi+1)
        profileclip = profile[slc]
        profileclip /= np.sum(np.maximum(profileclip,0))  # normalize
        yclip = y[slc]
        pars,perror = profilefit(yclip,profileclip)
        xmnarr.append(xmn)
        ymidarr.append(pars[1])
        ysigarr.append(pars[2])
        # Remember
        lasty = pars[1]
        lastsig = pars[2]
        
    # Backwards
    lasty = ymidarr[0]
    lastsig = ysigarr[0]
    for i in np.arange(1,nsteps):
        xmn = nx//2 - i*step
        xlo = xmn - nbin//2
        xhi = xmn + nbin//2 + 1        
        profile = np.nanmedian(im[:,xlo:xhi],axis=1)
        ind = np.argmax(profile)
        ylo = int(np.floor(lasty-2.5*lastsig))
        yhi = int(np.ceil(lasty+2.5*lastsig))
        slc = slice(ylo,yhi+1)
        profileclip = profile[slc]
        profileclip /= np.sum(np.maximum(profileclip,0))  # normalize
        yclip = y[slc]
        pars,perror = profilefit(yclip,profileclip)
        xmnarr.append(xmn)
        ymidarr.append(pars[1])
        ysigarr.append(pars[2])
        # Remember
        lasty = pars[1]
        lastsig = pars[2]

    #xmnarr = np.array(xmnarr)
    #ymidarr = np.array(ymidarr)
    #ysigarr = np.array(ysigarr)
    ttab = Table((xmnarr,ymidarr,ysigarr),names=['x','y','ysig'])
    ttab.sort('x')
        
    return ttab
        
    
def extract_optimal(im,ytrace,imerr=None,verbose=False,off=10,backoff=50,smlen=31):
    """ Extract a spectrum using optimal extraction (Horne 1986)"""
    ny,nx = im.shape
    yest = np.median(ytrace)
    # Get the subo,age
    yblo = int(np.maximum(yest-backoff,0))
    ybhi = int(np.minimum(yest+backoff,ny))
    nback = ybhi-yblo
    # Background subtract
    med = np.median(im[yblo:ybhi,:],axis=0)
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
    tot = np.sum(np.maximum(sim,0),axis=0)
    tot[(tot<=0) | ~np.isfinite(tot)] = 1
    psf1 = np.maximum(sim,0)/tot
    psf = np.zeros(psf1.shape,float)
    for i in range(nback):
        psf[i,:] = utils.medfilt(psf1[i,:],smlen)
        #psf[i,:] = utils.gsmooth(psf1[i,:],smlen)        
    psf[(psf<0) | ~np.isfinite(psf)] = 0
    totpsf = np.sum(psf,axis=0)
    totpsf[(totpsf<=0) | (~np.isfinite(totpsf))] = 1
    psf /= totpsf
    psf[(psf<0) | ~np.isfinite(psf)] = 0
    # Compute the weights
    wt = psf**2/serr**2
    wt[(wt<0) | ~np.isfinite(wt)] = 0
    totwt = np.sum(wt,axis=0)
    badcol = (totwt<=0)
    totwt[badcol] = 1
    # Compute the flux and flux error
    flux = np.sum(psf*sim/serr**2,axis=0)/totwt
    fluxerr = np.sqrt(1/totwt)    
    fluxerr[badcol] = 1e30  # bad columns
    # Recompute the trace
    trace = np.sum(psf*yy,axis=0)+yblo
    
    # Check for outliers
    diff = (sim-flux*psf)/serr**2
    bad = (diff > 25)
    if np.sum(bad)>0:
        # Mask bad pixels
        sim[bad] = 0
        serr[bad] = 1e20
        # Recompute the flux
        wt = psf**2/serr**2
        totwt = np.sum(wt,axis=0)
        badcol = (totwt<=0)
        totwt[badcol] = 1        
        flux = np.sum(psf*sim/serr**2,axis=0)/totwt
        fluxerr = np.sqrt(1/totwt)
        fluxerr[badcol] = 1e30  # bad columns
        # Recompute the trace
        trace = np.sum(psf*yy,axis=0)+yblo
        
    return flux,fluxerr,trace


def extract1d(filename):
    """ This performs 1D-extraction of the spectra."""

    # Load the calibrated file
    if os.path.exists(filename)==False:
        raise ValueError(filename+' NOT FOUND')
    print('Loading '+filename)
    input_model = datamodels.open(filename)
    print('data is '+str(type(input_model)))
    nslits = len(input_model.slits)
    print(str(nslits)+' slits')

    # Get the reference file
    print('Getting reference file')
    step = Extract1dStep()
    extract_ref = step.get_reference_file(input_model,'extract1d')
    print('Using reference file: '+extract_ref)
    extract_ref_dict = extract.open_extract1d_ref(extract_ref, input_model.meta.exposure.type)

        
    # Looping over slits
    for i in range(nslits):
        slit = input_model.slits[i]
        print('Slit ',i+1)
        print('source_name:',slit.source_name)
        print('source_id:',slit.source_id)
        print('slitlet_id:',slit.slitlet_id)
        print('source ra/dec:',slit.source_ra,slit.source_dec)
        
        # Get the data
        im = slit.data
        err = slit.err
        wave = slit.wavelength
        ny,nx = im.shape

        # Get extraction parameters, from extract.create_extraction()
        slitname = slit.name
        sp_order = extract.get_spectral_order(slit)        
        meta_source = slit
        smoothing_length = None
        use_source_posn = True
        extract_params = extract.get_extract_parameters(extract_ref_dict,meta_source,slitname,
                                                        sp_order,input_model.meta,step.smoothing_length,
                                                        step.bkg_fit,step.bkg_order,use_source_posn)
        extract_params['dispaxis'] = extract_ref_dict['apertures'][0]['dispaxis']

        # Get extraction model, extract.extract_one_slit()
        # If there is an extract1d reference file (there doesn't have to be), it's in JSON format.
        extract_model = extract.ExtractModel(input_model=input_model, slit=slit, **extract_params)
        ap = extract.get_aperture(im.shape, extract_model.wcs, extract_params)
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
        if extract_model.dispaxis == extract.HORIZONTAL:
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


        # Polynomial coefficients for traces
        # y(1024) = 0
        coef0 = np.array([ 4.59440169e-06, -1.86120908e-04, -4.83294422e+00])

        # Get peak at X=1024
        tot = np.nansum(np.maximum(im[:,974:1075],0),axis=1)
        yind = np.argmax(tot)
        # Trace for this star
        coef = coef0.copy()
        coef[2] += yind
        x = np.arange(2048)
        ytrace = np.polyval(coef,x)

        # Get the trace
        ttab = tracing(im,yind)
        tcoef = robust.polyfit(ttab['x'],ttab['y'],2)
        ytrace = np.polyval(tcoef,x)
        tsigcoef = robust.polyfit(ttab['x'],ttab['ysig'],1)
        ysig = np.polyval(tsigcoef,x)
        
        # Create the mask
        ybin = 3
        yy = np.arange(ny).reshape(-1,1) + np.zeros(nx).reshape(1,-1)
        mask = ((yy >= (ytrace-ybin)) & (yy <= (ytrace+ybin)))

        # I think the trace is at
        # ny/2+offset at Y=1024

        
        # Build PSF model

        # Boxcar
        boxsum = np.nansum(mask*im,axis=0)

        # Optimal extraction
        flux,fluxerr,trace = extract_optimal(im*mask,ytrace,imerr=err*mask,verbose=verbose,
                                             off=10,backoff=50,smlen=31):
        # Get the wavelengths
        wav = np.nansum(wave*mask,axis=0)/np.sum(mask,axis=0) * 1e4  # convert to Angstroms
        
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
        spec = Table((newwav,flux,fluxerr,trace),names=['wave','flux','flux_error','ytrace'])
        
        
    import pdb; pdb.set_trace()
