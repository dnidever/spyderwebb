# Packages that allow us to get information about objects:
import asdf
import os
import numpy as np
from astropy.table import Table
from dlnpyutils import utils as dln,robust
from dlnpyutils import plotting as pl
from glob import glob

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

def fluxcorr(files):
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
    wave = np.polyval(wcoef,np.arange(npix))

    # Load the spectra
    print('Loading the spectra')
    splist=[]
    for f in files:
        if os.path.getsize(f)>0:
            sp = doppler.read(f)
            splist.append(sp)
    nspec = len(splist)


    import pdb; pdb.set_trace()
    

    
    # Interpolate onto final grid
    spec = np.ones((nspec,npix),float)
    num = np.zeros((nspec,npix),int)
    for i in range(len(splist)):
        sp = splist[i].copy()
        if sp.ndim==2:
            for o in range(2):
                gdf, = np.where((wave>=np.min(sp.wave[:,o])) & (wave<=np.max(sp.wave[:,o])))
                gd, = np.where(np.isfinite(sp.flux[:,o]) & np.isfinite(sp.wave[:,o]) &
                               (sp.mask[:,o]==False))
                flux1 = dln.interp(sp.wave[gd,o],sp.flux[gd,o],wave[gdf])
                coef1 = robust.polyfit(wave[gdf],flux1,1)
                cont1 = np.polyval(coef1,wave[gdf])
                spec[i,gd] = flux1/cont1
                num[i,gd] += 1
        else:
                gdf, = np.where((wave>=np.min(sp.wave)) & (wave<=np.max(sp.wave)))
                gd, = np.where(np.isfinite(sp.flux) & np.isfinite(sp.wave) &
                               (sp.mask==False))
                flux1 = dln.interp(sp.wave[gd],sp.flux[gd],wave[gdf])
                coef1 = robust.polyfit(wave[gdf],flux1,1)
                cont1 = np.polyval(coef1,wave[gdf])
                spec[i,gd] = flux1/cont1
                num[i,gd] += 1

    # Throw out spectra with large scatter
                
    # Now smooth the spectra
                
    import pdb; pdb.set_trace()
                
            
    smlist=[]
    for i in range(len(splist)):
        sp = splist[i].copy()
        if sp.ndim==2:
            sp.flux = dln.medfilt(sp.flux[:,1],51)
            sp.wave = sp.wave[:,1]
            sp.err = sp.err[:,1]
            sp.mask = sp.mask[:,1]
            sp.ndim = 1
            smlist.append(sp)

    arr = np.zeros((len(smlist),2048))
    cnt = 0
    for i in range(len(smlist)):
        sp = smlist[i]
        cont0 = np.nanmedian(sp.flux)
        x = np.arange(sp.npix)
        coef = robust.polyfit(x,sp.flux,1)
        cont = np.polyval(coef,x)
        flux2 = sp.flux/cont
        sig = dln.mad(flux2)
        std = np.std(flux2)    
        if std<0.06:
            pl.oplot(flux2-1+i*0.02)
            arr[cnt,0:len(flux2)] = flux2
            cnt += 1
        print(i,sig)
    arr = arr[0:cnt,:]
            

    # USE FLUXED CANNON MODELS to get an absolute calibration

    
    # Step 1.  Calculate the relative flux correction
    #            using all the spectra


    
    # Step 2.  Apply the correction and run Doppler
    #            to get the best-fitting models for all
    #            spectra.  This is needed to remove
    #            real spectral features

    # Step 3.  Divide the spectra by the best-fitting models
    #            and recalculate the relative flux correction
    
