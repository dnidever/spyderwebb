# Packages that allow us to get information about objects:
import asdf
import os
import numpy as np
from astropy.table import Table
from dlnpyutils import utils as dln,robust
from dlnpyutils import plotting as pl
from glob import glob
from scipy.optimize import curve_fit

# Astropy tools:
from astropy.io import fits
from doppler.spec1d import Spec1D
from scipy.ndimage import median_filter,generic_filter
from scipy.signal import argrelextrema
import doppler
from . import utils

import matplotlib
import matplotlib.pyplot as plt


# Ignore these warnings
import warnings
warnings.filterwarnings("ignore", message="OptimizeWarning: Covariance of the parameters could not be estimated")

def nanmedfilt(x,size,mode='reflect'):
    return generic_filter(x, np.nanmedian, size=size)

def relfluxcorr(spectra):
    """ Calculate the relative flux correction given spectra."""

    import pdb; pdb.set_trace()

def fluxcorr(files,outfile=None):
    """ Calculate the relative flux correction."""

    # Get the files
    files = glob(files)
    # if a single file, check if it's a list
    if len(files)==1 and utils.is_binaryfile(files[0])==False:
        listfile = files[0]
        print('Reading list file:',listfile)
        files = dln.readlines(listfile)
    if len(files)==0:
        print('No files found')
        return

    #print('KLUDGE!!!!!!!!!')
    #files = files[0:10]
    
    # Only keep files that exist and have data
    files = [f for f in files if (os.path.exists(f) and (os.path.getsize(f)>0))]
    nfiles = len(files)
    if nfiles==0:
        print('No files found that exist and have data')
        return
    print(nfiles,' files')
        
    # wavelength coefficients with linear wavelength steps
    # 3834 pixels, from 9799.765 to 18797.7624 A
    wcoef = np.array([-1.35698061e-09, -7.79636391e-06,  2.39732634e+00,  9.79971041e+03])
    npix = 3834
    xpix = np.arange(npix)
    wave = np.polyval(wcoef,xpix)

    # Load the spectra
    print('Loading the spectra')
    splist=[]
    for f in files:
        if os.path.getsize(f)>0:
            sp = doppler.read(f)
            splist.append(sp)
    nspec = len(splist)

    #------------------------------------------------    
    # Step 1.  Calculate the relative flux correction
    #            using all the spectra

    
    # Interpolate onto final grid
    spec = np.ones((nspec,npix),float)
    err = np.ones((nspec,npix),float)    
    num = np.zeros((nspec,npix),int)
    for i in range(nspec):
        sp = splist[i].copy()
        if sp.ndim==2:
            for o in range(2):
                gdf, = np.where((wave>=np.nanmin(sp.wave[:,o])) & (wave<=np.nanmax(sp.wave[:,o])))
                gd, = np.where(np.isfinite(sp.flux[:,o]) & np.isfinite(sp.wave[:,o]) &
                               (sp.mask[:,o]==False))
                if len(gd)==0 or len(gdf)==0:
                    import pdb; pdb.set_trace()
                flux1 = dln.interp(sp.wave[gd,o],sp.flux[gd,o],wave[gdf])
                err1 = dln.interp(sp.wave[gd,o],sp.err[gd,o],wave[gdf])
                # Get linear continuum fit
                coef1 = robust.polyfit(wave[gdf],flux1,1)
                cont1 = np.polyval(coef1,wave[gdf])
                # Save the normalized spectrum
                spec[i,gdf] = flux1/cont1
                err[i,gdf] = err1/cont1
                num[i,gdf] += 1
        else:
                gdf, = np.where((wave>=np.nanmin(sp.wave)) & (wave<=np.nanmax(sp.wave)))
                gd, = np.where(np.isfinite(sp.flux) & np.isfinite(sp.wave) &
                               (sp.mask==False))
                if len(gd)==0 or len(gdf)==0:
                    import pdb; pdb.set_trace()                
                flux1 = dln.interp(sp.wave[gd],sp.flux[gd],wave[gdf])
                err1 = dln.interp(sp.wave[gd],sp.err[gd],wave[gdf])
                # Get linear continuum fit
                coef1 = robust.polyfit(wave[gdf],flux1,1)
                cont1 = np.polyval(coef1,wave[gdf])
                # Save the normalized spectrum
                spec[i,gdf] = flux1/cont1
                err[i,gdf] = err1/cont1                
                num[i,gdf] += 1
                
    # Now smooth the spectra
    smspec = np.copy(spec)
    for i in range(nspec):
        #smspec[i,:] = dln.medfilt(spec[i,:],51)
        smspec[i,:] = dln.gsmooth(median_filter(spec[i,:],11),40)        

    # Throw out spectra with large scatter
    specsigma = np.zeros(nspec)+99999.
    for i in range(nspec):
        spec1 = smspec[i,:]
        err1 = err[i,:]
        gd, = np.where(np.isfinite(spec1) & np.isfinite(err1) & (err1<1e20))
        std = np.std(spec1[gd])
        sig = dln.mad(spec1[gd])
        specsigma[i] = std
    medspecsigma = np.nanmedian(specsigma)
    sigspecsigma = dln.mad(specsigma)
    gdspec, = np.where(specsigma < (medspecsigma+3*sigspecsigma))
    print('Keeping ',len(gdspec),' of ',nspec,' spectra')
    smspec = smspec[gdspec,:]
    spec = spec[gdspec,:]    
    err = err[gdspec,:]
    num = num[gdspec,:]
    nspec = len(gdspec)
    
    # Calculate the median relative offset
    reloff1 = np.nanmean(smspec,axis=0)

    #diff = np.zeros(spec.shape)
    #for i in range(nspec):
    #    diff[i,:] = smspec[i,:]-reloff
    #sig = dln.mad(diff,axis=1)
    #medsig = np.median(sig)
    #sigsig = dln.mad(sig)
    #gdspec2, = np.where(sig < medsig+3*sigsig)
    #smspec = smspec[gdspec2,:]
    #spec = spec[gdspec2,:]    
    #err = err[gdspec2,:]
    #num = num[gdspec2,:]
    #nspec = len(gdspec2)

     
    # Calculate the median relative offset
    #reloff2 = np.nanmean(smspec,axis=0)
    
    
    # fix edges
    #reloff[0:40]=0.99
    #reloff[-15:]=0.99
    
    #pl.plot(wave,reloff,yr=[0.9,1.1])
    ##pl.oplot(wave,1.0+0.03*np.sin(wave/100),c='r')
    ##pl.oplot(wave,1.0+0.015*np.sin(wave/200),c='green')
    #pl.oplot(wave,1.0+0.015*np.sin(wave/130+2.0),c='green')

    # sin(2*pi*x/lambda+phi)

    #def wavefunc(x,amp,wavelength,phi,const):
    #    return amp*np.sin(2*np.pi*x/wavelength+phi)+const
    #initpar = [0.015,820.0,2.0,1.0]
    #pars,pcov = curve_fit(wavefunc,wave,reloff,p0=initpar)
    #mm = wavefunc(wave,*pars)

    # array([ 3.72375182e-03,  7.87562008e+02, -1.25323776e+00,  1.00054443e+00])
    # array([7.45125964e-03, 8.81892624e+02, 5.23014812e+00, 9.99417155e-01])

    #import pdb; pdb.set_trace()
    
    #return wave,reloff
    
    
    # USE FLUXED CANNON MODELS to get an absolute calibration


    #---------------------------------------------------    
    # Step 2.  Apply the correction and run Doppler
    #            to get the best-fitting models for all
    #            spectra.  This is needed to remove
    #            real spectral features
    print('Running Doppler on all spectra')

    # Doppler Cannon model only covers up to 18000
    #bad = (sp.wave >= 18000)
    #if np.sum(bad)>0:
    #    sp.flux[bad] = 0.0
    #    sp.err[bad] = 1e30
    #    sp.wave[bad] = 0.0
    #    sp.mask[bad] = True

    # Loop over spectra and run Doppler
    bestmodels = []
    dt = [('vhelio',float),('vrel',float),('vrelerr',float),('teff',float),('tefferr',float),
          ('logg',float),('loggerr',float),('feh',float),('feherr',float),('chisq',float),('bc',float)]
    out = np.zeros(len(splist),dtype=np.dtype(dt))
    for i in range(len(splist)):
        sp = splist[i].copy()
        
        # Divide by the initial flux correction factor
        for j,sp1 in enumerate(sp):
            fluxcorr1 = dln.interp(wave,reloff1,sp1.wave)
            sp1.flux /= fluxcorr1
            sp1.err /= fluxcorr1
        
        # Doppler Cannon model only covers up to 18000
        bad = (sp.wave >= 17900)
        if np.sum(bad)>0:
            sp.flux[bad] = 0.0
            sp.err[bad] = 1e30
            sp.wave[bad] = 0.0
            sp.mask[bad] = True
        #    import pdb; pdb.set_trace()
        
        try:
            out1,bestmodel,spm = doppler.fit(sp,verbose=False)
            print(i+1,out1['teff'][0],out1['logg'][0],out1['feh'][0],out1['vhelio'][0])
            out[i] = out1
            bestmodels.append(bestmodel)
        except:
            print(i+1,'Doppler failed')
            bestmodels.append(None)

            
    #-------------------------------------------------------    
    # Step 3.  Divide the spectra by the best-fitting models
    #            and recalculate the relative flux correction
    
            
    # Interpolate best model onto the final grid and
    #   divide the observed spectra by it
    normspec = spec.copy()
    normerr = err.copy()    
    modelspec = np.zeros(spec.shape,float)
    for i in range(len(splist)):
        sp = splist[i].copy()        
        bestmod = bestmodels[i]
        if bestmod is None:
            continue
        for i,sp1 in enumerate(sp):
            bestmod1 = bestmod[i]
            gdf, = np.where((wave>=np.nanmin(sp1.wave)) & (wave<=np.nanmax(sp1.wave)))
            gd, = np.where(np.isfinite(sp1.flux) & np.isfinite(sp1.wave) &
                           (sp1.mask==False))
            if len(gd)==0 or len(gdf)==0:
                import pdb; pdb.set_trace()                
            flux1 = dln.interp(sp1.wave[gd],sp1.flux[gd],wave[gdf])
            err1 = dln.interp(sp1.wave[gd],sp1.err[gd],wave[gdf])            
            mgd, = np.where(np.isfinite(bestmod1.flux) & np.isfinite(bestmod1.wave) &
                           (bestmod1.mask==False))
            mflux1 = dln.interp(bestmod1.wave[mgd],bestmod1.flux[mgd],wave[gdf])
            normflux1 = flux1/mflux1
            normerr1 = err1/mflux1
            mednormflux1 = np.nanmedian(normflux1)
            normflux1 /= mednormflux1
            normerr1 /= mednormflux1
            # Save the normalized spectrum
            modelspec[i,gdf] = mflux1
            normspec[i,gdf] = normflux1
            normerr[i,gdf] = normerr1
            num[i,gdf] += 1
            

            
        #if sp.ndim==2:
        #    for o in range(2):
        #        gdf, = np.where((wave>=np.nanmin(sp.wave[:,o])) & (wave<=np.nanmax(sp.wave[:,o])))
        #        gd, = np.where(np.isfinite(sp.flux[:,o]) & np.isfinite(sp.wave[:,o]) &
        #                       (sp.mask[:,o]==False))
        #        if len(gd)==0 or len(gdf)==0:
        #            import pdb; pdb.set_trace()
        #        flux1 = dln.interp(sp.wave[gd,o],sp.flux[gd,o],wave[gdf])
        #        err1 = dln.interp(sp.wave[gd,o],sp.err[gd,o],wave[gdf])                
        #        mflux1 = dln.interp(bestmod.wave[gd,o],bestmod.flux[gd,o],wave[gdf])                
        #        normflux1 = flux1/mflux1
        #        normerr1 = err1/mflux1
        #        mednormflux1 = np.nanmedian(normflux1)
        #        normflux1 /= mednormflux1
        #        normerr1 /= mednormflux1
        #        # Get linear continuum fit
        #        #coef1 = robust.polyfit(wave[gdf],normflux1,1)
        #        #cont1 = np.polyval(coef1,wave[gdf])
        #        # Save the normalized spectrum
        #        modelspec[i,gdf] = mflux1
        #        normspec[i,gdf] = normflux1
        #        normerr[i,gdf] = normerr1
        #        num[i,gdf] += 1
        #else:
        #    gdf, = np.where((wave>=np.nanmin(sp.wave)) & (wave<=np.nanmax(sp.wave)))
        #    gd, = np.where(np.isfinite(sp.flux) & np.isfinite(sp.wave) &
        #                   (sp.mask==False))
        #    if len(gd)==0 or len(gdf)==0:
        #        import pdb; pdb.set_trace()                
        #    flux1 = dln.interp(sp.wave[gd],sp.flux[gd],wave[gdf])
        #    err1 = dln.interp(sp.wave[gd],sp.err[gd],wave[gdf])            
        #    mflux1 = dln.interp(bestmod.wave[gd],bestmod.flux[gd],wave[gdf])
        #    normflux1 = flux1/mflux1
        #    normerr1 = err1/mflux1
        #    mednormflux1 = np.nanmedian(normflux1)
        #    normflux1 /= mednormflux1
        #    normerr1 /= mednormflux1
        #    # Get linear continuum fit
        #    #coef1 = robust.polyfit(wave[gdf],normflux1,1)
        #    #cont1 = np.polyval(coef1,wave[gdf])
        #    # Save the normalized spectrum
        #    modelspec[i,gdf] = mflux1
        #    normspec[i,gdf] = normflux1
        #    normerr[i,gdf] = normerr1
        #    num[i,gdf] += 1

    gdspec, = np.where(np.nansum(normspec,axis=1) != 0)
    normspec = normspec[gdspec,:]
    normerr = normerr[gdspec,:]
    nspec = len(gdspec)
            
    # Now smooth the spectra
    smnormspec = np.zeros(normspec.shape,float)
    for i in range(nspec):
        smnormspec[i,:] = dln.gsmooth(median_filter(normspec[i,:],11),40)        

    # Throw out spectra with large scatter
    normspecsigma = np.zeros(nspec)+99999.
    for i in range(nspec):
        spec1 = smnormspec[i,:]
        err1 = normerr[i,:]
        gd, = np.where(np.isfinite(spec1) & np.isfinite(err1) & (err1<1e20))
        std = np.std(spec1[gd])
        sig = dln.mad(spec1[gd])
        normspecsigma[i] = std
    medspecsigma = np.nanmedian(normspecsigma)
    sigspecsigma = dln.mad(normspecsigma)
    gdspec, = np.where(normspecsigma < (medspecsigma+3*sigspecsigma))
    print('Keeping ',len(gdspec),' of ',nspec,' spectra')
    smnormspec = smnormspec[gdspec,:]
    normspec = normspec[gdspec,:]    
    normerr = normerr[gdspec,:]
    #num = num[gdspec,:]
    nspec = len(gdspec)
    
    # Calculate the median relative offset
    reloff = np.nanmean(smspec,axis=0)

            
    import pdb; pdb.set_trace()

    # Save to output file
    if outfile is None:
        outfile = 'nirspec_fluxcorr.fits'
    print('Writing flux correction to ',outfile)
    hdu = fits.PrimaryHDU(reloff)
    hdu.writeto(outfile,overwrite=True)
    hdu.close()

