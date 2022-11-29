# Packages that allow us to get information about objects:
import asdf
import os
import numpy as np
from astropy.table import Table
from dlnpyutils import utils as dln,robust

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
from jwst.extract_1d import extract as jextract

from scipy.ndimage.filters import median_filter, gaussian_filter

from glob import glob
import doppler
from doppler.spec1d import Spec1D
from . import extract, utils, sincint


def getexpinfo(obsname,logger=None):

    #if logger is None: logger=dln.basiclogger()
    
    print('Looking for data for obsname='+obsname)
    dirs = glob(obsname+'*')
    # Only want directories
    dirs = [d for d in dirs if os.path.isdir(d)]
    dirs = [d for d in dirs if (d.endswith('_nrs1') or d.endswith('_nrs2'))]
    if len(dirs)==0:
        raise ValueError('No directories found for '+obsname)
    
    # Get exposures names
    base = [d[:-5] for d in dirs]
    expnames = np.unique(base)
    nexp = len(expnames)
    print('Found '+str(nexp)+' exposures')
    
    # Check the _cal.fits files and get basic information
    edict = []
    for i in range(nexp):
        expname = expnames[i]
        calfile = expname+'_nrs1/'+expname+'_nrs1_cal.fits'
        if os.path.exists(calfile)==False:
            print(calfile+' NOT FOUND')
            continue
        hd = fits.getheader(calfile,0)
        hdict = {'FILENAME':calfile,'EXPNAME':expname}
        cards = ['OPMODE','EXP_TYPE','FILTER','GRATING','INSTRUMENT','DETECTOR','TARG_RA','TARG_DEC','VISITID',
                 'VISIT','EXPOSURE','OBSLABEL','NOD_TYPE','APERNAME','DATE-BEG']
        for c in cards: hdict[c]=hd.get(c)
        # Observation ID
        hdict['OBSID'] = hdict['GRATING']+'-'+hdict['FILTER']+'-'+hdict['OBSLABEL']
        
        # Only reduce exposures that are dispersed
        # Dispersed has
        #  EXP_TYPE= 'NRS_MSASPEC' 
        #  GRATING = 'G140H'        
        # Imaging has
        #  EXP_TYPE = 'NRS_MSATA'
        #  GRATING = 'MIRROR' 
        if hdict['GRATING']=='MIRROR':
            print('This is not a dispersed exposure.  Skipping')
            continue

        # Add to the list
        edict.append(hdict)

    return edict

def joinspec(sp1,sp2):
    """ Combine NRS1 and NRS2."""

    # Combine the two spectra
    #  use 2D Spec1D arrays
    npix = np.max([sp1.npix,sp2.npix])
    flux = np.zeros((npix,2),float)
    flux[0:sp1.npix,0] = sp1.flux
    flux[0:sp2.npix,1] = sp2.flux
    err = np.zeros((npix,2),float)+1e30
    err[0:sp1.npix,0] = sp1.err
    err[0:sp2.npix,1] = sp2.err
    mask = np.ones((npix,2),bool)
    mask[0:sp1.npix,0] = sp1.mask
    mask[0:sp2.npix,1] = sp2.mask
    wave = np.zeros((npix,2),float)
    wave[0:sp1.npix,0] = sp1.wave
    wave[0:sp2.npix,1] = sp2.wave        
    # Put it all together
    sp = Spec1D(flux,err=err,wave=wave,mask=mask,instrument='NIRSpec')
    sp.source_name = sp1.source_name
    sp.source_id = sp1.source_id
    sp.slitlet_id = sp1.slitlet_id
    sp.source_ra = sp1.source_ra
    sp.source_dec = sp1.source_dec
    sp.ytrace1 = sp1.ytrace
    sp.xstart1 = sp1.xstart
    sp.xsize1 = sp1.xsize
    sp.ystart1 = sp1.ystart
    sp.yize1 = sp1.ysize
    sp.offset = sp1.offset
    sp.tcoef1 = sp1.tcoef
    sp.tsigcoef1 = sp1.tsigcoef
    sp.ytrace2 = sp2.ytrace
    sp.xstart2 = sp2.xstart
    sp.xsize2 = sp2.xsize
    sp.ystart2 = sp2.ystart
    sp.yize2 = sp2.ysize
    sp.tcoef2 = sp2.tcoef
    sp.tsigcoef2 = sp2.tsigcoef

    return sp

def stackspec(splist):
    """ Stack multiple spectra of the same source."""

    nspec = len(splist)
    npix = splist[0].npix
    norder = splist[0].norder

    # The two detectors are about 147 pixels apart

    # Array of sinc widths
    nres = [3.5,3.5]

    # wavelength coefficients with linear wavelength steps
    # 3834 pixels, from 9799.765 to 18797.7624 A
    wcoef = np.array([-1.35698061e-09, -7.79636391e-06,  2.39732634e+00,  9.79971041e+03])
    nfpix = 3834
    fwave = np.polyval(wcoef,np.arange(nfpix))
    
    # initialize array for stack of interpolated spectra
    zeros = np.zeros([nspec,nfpix])
    izeros = np.zeros([nspec,nfpix],bool)
    stack = Spec1D(np.zeros(10),wave=np.zeros(10))
    stack.flux = zeros
    stack.err = zeros.copy()
    stack.wave = fwave
    stack.mask = np.ones([nspec,nfpix],bool)
    stack.cont = zeros.copy()
    
        #gdwave, = np.where(fwave > 0)
        #fwave = fwave[gdwave]
        
    # Loop over each exposure and interpolate to final wavelength grid
    for i in range(nspec):
        spec = splist[i]

        # Loop over the detectors/orders
        for o in range(norder):

            # Get the good pixels
            if spec.ndim==2:
                gdpix, = np.where(spec.wave[:,o] > 0)
                ngdpix = len(gdpix)
                wave = spec.wave[gdpix,o]
                flux = spec.flux[gdpix,o]
                err = spec.err[gdpix,o]
                mask = spec.mask[gdpix,o]
            else:
                gdpix, = np.where(spec.wave > 0)
                ngdpix = len(gdpix)
                wave = spec.wave[gdpix]
                flux = spec.flux[gdpix]
                err = spec.err[gdpix]
                mask = spec.mask[gdpix]
                
            # Get the pixel values to interpolate to
            #pix = utils.wave2pix(wave,fwave)
            #gd, = np.where(np.isfinite(pix))
            
            # Get a smoothed, filtered spectrum to use as replacement for bad values
            msmlen = np.minimum(501,ngdpix//2)
            if msmlen % 2 ==0: msmlen+=1  # want odd
            gsmlen = np.minimum(100,ngdpix//4)
            
            cont = gaussian_filter(median_filter(flux,[msmlen],mode='reflect'),gsmlen)
            # Deal with super high error values for bad pixels
            bderr, = np.where((mask==True) | (err>1e20))
            gderr, = np.where((mask==False) & (err<1e20))
            temperr = err.copy()
            if len(bderr)>0:
                temperr[bderr] = np.median(err[gderr])
            errcont = gaussian_filter(median_filter(temperr,[msmlen],mode='reflect'),gsmlen)
            bad, = np.where(mask | (err>1e20))
            if len(bad) > 0:
                flux[bad] = cont[bad]
                err[bad] = errcont[bad]
                mask[bad] = True
                
            # Load up quantity/error pairs for interpolation
            raw = [[flux,err**2],
                   [mask.astype(float),None]]

            # Do the sinc interpolation
            # sincint(x,nres,speclist)
            # x is desired positions
            #out = sincint.sincint(pix[gd],nres[o],raw)
            #newflux = out[0][0]
            #newerr = out[0][1]
            #newmask = out[1][0]

            gd, = np.where((fwave >= np.min(wave)) & (fwave <= np.max(wave)))
            newflux = dln.interp(wave,flux,fwave[gd],kind='cubic',extrapolate=False)
            newerr = dln.interp(wave,err,fwave[gd],kind='cubic',extrapolate=False)
            newmask = dln.interp(wave,mask.astype(float),fwave[gd],kind='cubic',extrapolate=False)            
            #gd, = np.where(np.isfinite(newflux))
            
            # From output flux, get continuum to remove, so that all spectra are
            #   on same scale. We'll later multiply in the median continuum
            #newflux = out[0][0]
            stack.cont[i,gd] = gaussian_filter(median_filter(newflux,[msmlen],mode='reflect'),gsmlen)

            # Load interpolated spectra into output stack
            #stack.wave[0:len(fwave)] = fwave
            stack.flux[i,gd] += newflux / stack.cont[i,gd]
            stack.err[i,gd] += newerr / stack.cont[i,gd]
            # For mask, set bits where interpolated value is below some threshold to "good"
            goodmask, = np.where(newmask < 0.5)
            if len(goodmask)>0:
                stack.mask[i,gd[goodmask]] = False

            # Set ERR of bad pixels to 1e30
            badpix, = np.where(stack.mask[i,:])
            if len(badpix)>0:
                stack.err[i,badpix] = 1e30
                
    # Create final spectrum
    zeros = np.zeros(splist[0].flux.shape)
    izeros = np.zeros(splist[0].flux.shape,bool)
    comb = Spec1D(zeros,err=zeros.copy(),mask=np.ones(splist[0].flux.shape,bool),wave=stack.wave.copy())
    comb.cont = zeros.copy()
    
    # Pixel-by-pixel weighted average
    cont = np.mean(stack.cont,axis=0)
    comb.flux = np.sum(stack.flux/stack.err**2,axis=0)/np.sum(1./stack.err**2,axis=0) * cont
    comb.err =  np.sqrt(1./np.sum(1./stack.err**2,axis=0)) * cont
    comb.mask = np.bitwise_and.reduce(stack.mask,0)
    comb.cont = cont
        
    return comb,stack


def reduce(obsname,outdir='./',logger=None,clobber=False):
    """ This reduces the JWST NIRSpec MSA data """

    if outdir.endswith('/')==False: outdir+='/'
    #if logger is None: logger=dln.basiclogger()
    stackdir = outdir+'stack/'
    if os.path.exists(stackdir)==False: os.makedirs(stackdir)
    
    # Get exposures information
    edict = getexpinfo(obsname)
    nexp = len(edict)
    if nexp==0:
        print('No exposures to reduce')
        return None
    print(str(nexp)+' dispersed exposures')

    # Group the exposures in grating/filter/target
    obsids = np.array([h['OBSID'] for h in edict])
    uobsids = np.unique(obsids)
    print('Found '+str(len(uobsids))+' observation group(s): '+', '.join(uobsids))

    # Loop over observations groups
    for o in range(len(uobsids)):
        obsid = uobsids[o]
        odir = outdir+obsid+'/'
        if os.path.exists(odir)==False:
            os.makedirs(odir)
        ind, = np.where(obsids==obsid)
        nind = len(ind)
        expnames = [edict[e]['EXPNAME'] for e in ind]
        
        print(' ')
        print('=============================================')
        print('Observation Group '+str(o+1)+': '+obsid)
        print(str(nind)+' exposures')
        print('Using output directory: '+odir)
        print('---------------------------------------------') 
        
        # Loop over exposures and extract the spectra
        expspec = []
        sourcenames = []
        for i in range(nexp):
            expname = expnames[i]
            print(' ')
            print('------------------------------------')            
            print('EXPOSURE '+str(i+1)+' '+expname)
            print('------------------------------------')
            speclist = extractexp(expname,outdir=odir,clobber=clobber)
            srcname = [s.source_name for s in speclist]
            expspec.append(speclist)
            sourcenames += srcname

        # Loop over sources
        print('Combining spectra')
        sourcenames = np.unique(np.array(sourcenames))
        nsources = len(sourcenames)
        for i in range(nsources):
            srcname = sourcenames[i]
            print(i+1,srcname)
            
            # Stack spectra from multiple exposures
            if nexp>1:
                # Loop over exposures
                splist = []
                for e in range(nexp):
                    especlist = expspec[e]  # list of all spectra from this exposures
                    esourcename = np.array([s.source_name for s in especlist])
                    ind, = np.where(esourcename==srcname)
                    if len(ind)>0:
                        splist.append(especlist[ind[0]])
                        
                # Do the stacking
                if len(splist)>1:
                    print('Combining spectra from multiple exposures')
                    combsp,stack = stackspec(splist)
                else:
                    combsp = splist[0]
            else:
                combsp = splist[0]
                
            # Write to file
            outfile = stackdir+'/'+srcname+'_stack.fits'
            print('Writing to '+outfile)
            combsp.write(outfile,overwrite=True)
                
            #import pdb; pdb.set_trace()
            
        import pdb; pdb.set_trace()



def extractexp(expname,logger=None,outdir='./',clobber=False):
    """ This performs 1D-extraction of the spectra in one exposure."""

    #if logger is None: logger=dln.basiclogger()

    # Load the calibrated file    
    filename1 = expname+'_nrs1/'+expname+'_nrs1_cal.fits'
    filename2 = expname+'_nrs2/'+expname+'_nrs2_cal.fits'
    if os.path.exists(filename1)==False:
        raise ValueError(filename1+' NOT FOUND')
    if os.path.exists(filename2)==False:
        raise ValueError(filename2+' NOT FOUND')

    hdu1 = fits.open(filename1)
    nsources1 = int(len(hdu1)-2/8)
    sourceid1 = []
    sourcename1 = []    
    for i in np.arange(1,len(hdu1)):
        if hdu1[i].header.get('extname')=='SCI':
            sourceid1.append(hdu1[i].header['srcalias'])
            sourcename1.append(hdu1[i].header['srcname']) 
    sourceid1 = np.array(sourceid1)
    hdu1.close()
    hdu2 = fits.open(filename2)
    sourceid2 = []
    sourcename2 = []
    for i in np.arange(1,len(hdu2)):
        if hdu2[i].header.get('extname')=='SCI':
            sourceid2.append(hdu2[i].header['srcalias'])
            sourcename2.append(hdu2[i].header['srcname'])
    sourceid2 = np.array(sourceid2)
    hdu2.close()
    allsourceids = np.hstack((sourceid1,sourceid2))
    allsourcenames = np.hstack((sourcename1,sourcename2))
    _,ui = np.unique(allsourceids,return_index=True)
    sourceids = allsourceids[ui]
    sourcenames = allsourcenames[ui]
    nsources = len(sourceids)
    print(str(nsources)+' sources')
    
    # Looping over sources
    data1 = None
    data2 = None
    speclist = []
    for i in range(nsources):
        sourceid = sourceids[i]
        sourcename = sourcenames[i]
        sp = None
        print(' ')
        print('--',i+1,sourceid,'--')
        
        # Check if it already exists
        outfile = outdir+sourcename+'_'+expname+'.fits'
        if os.path.exists(outfile) and clobber==False:
            if os.path.getsize(outfile)==0:
                print(outfile+' is an empty file.')
                continue
            print(outfile+' already exists. Loading')            
            sp = doppler.read(outfile)
            speclist.append(sp)
            continue

        if data1 is None:
            print('Loading '+filename1)
            data1 = datamodels.open(filename1)
        if data2 is None:
            print('Loading '+filename2)    
            data2 = datamodels.open(filename2)
        
        # NRS1
        sp1 = None
        ind1, = np.where(sourceid1==sourceid)
        if len(ind1)>0:
            sp1 = extract.extract_slit(data1,data1.slits[ind1[0]])
        # NRS2
        sp2 = None
        ind2, = np.where(sourceid2==sourceid)
        if len(ind2)>0:
            sp2 = extract.extract_slit(data2,data2.slits[ind2[0]])

        # Join the two spectra together
        if sp1 is not None and sp2 is not None:
            sp = joinspec(sp1,sp2)
        else:
            if sp1 is not None: sp=sp1
            if sp2 is not None: sp=sp2            
            
        # Save the file
        if sp is not None:
            print('Writing to '+outfile)            
            sp.write(outfile,overwrite=True)
            speclist.append(sp)
        else:
            dln.touch(outfile)
            
    # Close the files
    if data1 is not None: data1.close()
    if data2 is not None: data2.close()
        
    return speclist
