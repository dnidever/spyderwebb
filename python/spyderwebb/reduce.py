# Packages that allow us to get information about objects:
import asdf
import os
import numpy as np
from astropy.table import Table
from dlnpyutils import utils as dln,robust
import tempfile
import shutil
import traceback

# Astropy tools:
from astropy.io import fits

import jwst

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
from . import extract, utils, sincint, qa

import matplotlib
import matplotlib.pyplot as plt
from dlnpyutils import plotting as pl


def getexpinfo(obsname,logger=None,redtag='red'):

    #if logger is None: logger=dln.basiclogger()
    
    print('Looking for data for obsname='+obsname)
    files = glob(obsname+'*'+redtag+'*')
    nfiles = len(files)
    if len(files)==0:
        raise ValueError('No files found for '+obsname)    
    #dirs = glob(obsname+'*')
    ## Only want directories
    #dirs = [d for d in dirs if os.path.isdir(d)]
    #dirs = [d for d in dirs if (d.endswith('_nrs1') or d.endswith('_nrs2'))]
    #if len(dirs)==0:
    #    raise ValueError('No directories found for '+obsname)
    
    # Get exposures names
    #base = [d[:-5] for d in dirs]
    base = [os.path.basename(f) for f in files]
    base = [b[:b.find('_nrs')] for b in base]
    expnames = np.unique(base)
    nexp = len(expnames)
    print('Found '+str(nexp)+' exposures')
    
    # Check the _red.fits files and get basic information
    edict = []
    for i in range(nexp):
        expname = expnames[i]
        #calfile = expname+'_nrs1/'+expname+'_nrs1_cal.fits'
        calfile = expname+'_nrs1_'+redtag+'.fits'
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
    if hasattr(sp1,'fluxcorr'):
        fluxcorr = np.ones((npix,2),float)
        fluxcorr[0:sp1.npix,0] = sp1.fluxcorr
        fluxcorr[0:sp2.npix,1] = sp2.fluxcorr
 
    # Combine the LSF parameters
    # x-type is wavelengths
    nlpars = np.max((len(sp1.lsf.pars[:,0]),len(sp2.lsf.pars[:,0])))
    lsfpars = np.zeros((nlpars,2),float)
    lsfpars[0:len(sp1.lsf.pars[:,0]),0] = sp1.lsf.pars[:,0]
    lsfpars[0:len(sp2.lsf.pars[:,0]),1] = sp2.lsf.pars[:,0]
    #lsfpars = np.hstack((sp1.lsf.pars,sp2.lsf.pars))
    lsftype = sp1.lsf.lsftype
    lsfxtype = sp1.lsf.xtype
    # Put it all together
    sp = Spec1D(flux,err=err,wave=wave,mask=mask,instrument='NIRSpec',
                lsfpars=lsfpars,lsftype=lsftype,lsfxtype=lsfxtype)
    sp.jd = sp1.jd
    if hasattr(sp1,'fluxcorr'):
        sp.fluxcorr = fluxcorr
    sp.exptime = sp1.exptime
    sp.bc = sp1.bc
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


    # Need to rescale NRS1
    #  median is 1.096
    rescale_nrs1 = 1.096
    
    # The two detectors are about 147 pixels apart

    # Array of sinc widths
    nres = [3.5,3.5]

    # wavelength coefficients with linear wavelength steps
    # 3834 pixels, from 9799.765 to 18797.7624 A
    wcoef = np.array([-1.35698061e-09, -7.79636391e-06,  2.39732634e+00,  9.79971041e+03])
    nfpix = 3834
    fwave = np.polyval(wcoef,np.arange(nfpix))

    hasfluxcorr = hasattr(splist[0],'fluxcorr')
    
    # initialize array for stack of interpolated spectra
    zeros = np.zeros([nspec,nfpix])
    izeros = np.zeros([nspec,nfpix],bool)
    stack = Spec1D(np.ones(10),wave=np.arange(10))
    stack.flux = zeros
    stack.err = zeros.copy()+1e30
    stack.wave = fwave
    stack.mask = np.ones([nspec,nfpix],bool)
    if hasfluxcorr:
        stack.fluxcorr = np.ones([nspec,nfpix])
    stack.cont = zeros.copy()
    stack.bc = np.zeros(nspec,float)
    lsfwsigma = np.zeros([nspec,nfpix],bool)+np.nan

    # Loop over each exposure and interpolate to final wavelength grid
    for i in range(nspec):
        spec = splist[i]

        # Barycentric correction
        if spec.bc is None:
            bc = spec.barycorr()
        else:
            bc = spec.bc
        stack.bc[i] = bc
        
        # Loop over the detectors/orders
        for o in range(spec.norder):

            # Get the good pixels
            if spec.ndim==2:
                gdpix, = np.where(spec.wave[:,o] > 0)
                ngdpix = len(gdpix)
                wave = spec.wave[gdpix,o]
                flux = spec.flux[gdpix,o]
                err = spec.err[gdpix,o]
                mask = spec.mask[gdpix,o]
                lsfpars = np.atleast_2d(spec.lsf.pars)[:,o]
                if hasfluxcorr:
                    fluxcorr = spec.fluxcorr[gdpix,o]
            else:
                gdpix, = np.where(spec.wave > 0)
                ngdpix = len(gdpix)
                wave = spec.wave[gdpix]
                flux = spec.flux[gdpix]
                err = spec.err[gdpix]
                mask = spec.mask[gdpix]
                lsfpars = np.atleast_2d(spec.lsf.pars)[:,o]
                if hasfluxcorr:
                    fluxcorr = spec.fluxcorr[gdpix]

            # Rescale NRS1 to NRS2 levels
            if spec.ndim==2 and o==0:
                flux *= rescale_nrs1
                err *= rescale_nrs1

            # Mask Nan/Inf pixels
            bdpix, = np.where((~np.isfinite(flux)) | (~np.isfinite(err)))
            if len(bdpix)>0:
                flux[bdpix] = 0.0
                err[bdpix] = 1e30
                mask[bdpix] = True
                                
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
            if hasfluxcorr:
                newfluxcorr = dln.interp(wave,fluxcorr,fwave[gd],kind='cubic',extrapolate=False)                        
            #gd, = np.where(np.isfinite(newflux))
            
            # From output flux, get continuum to remove, so that all spectra are
            #   on same scale. We'll later multiply in the median continuum
            #newflux = out[0][0]
            stack.cont[i,gd] = gaussian_filter(median_filter(newflux,[msmlen],mode='reflect'),gsmlen)
            
            # Load interpolated spectra into output stack
            #stack.wave[0:len(fwave)] = fwave
            stack.flux[i,gd] = newflux / stack.cont[i,gd]
            stack.err[i,gd] = newerr / stack.cont[i,gd]
            if hasfluxcorr:
                stack.fluxcorr[i,gd] = newfluxcorr
            # For mask, set bits where interpolated value is below some threshold to "good"
            goodmask, = np.where(newmask < 0.5)
            if len(goodmask)>0:
                stack.mask[i,gd[goodmask]] = False

            # Set ERR of bad pixels to 1e30
            badpix, = np.where(stack.mask[i,:])
            if len(badpix)>0:
                stack.err[i,badpix] = 1e30

            # LSF sigma
            if spec.lsf.xtype.lower()=='wave':
                lwsig = np.polyval(lsfpars[::-1],fwave[gd])
            else:
                import pdb; pdb.set_trace()
            lsfwsigma[i,gd] = lwsig
            
    # Combine LSF, take average of the LSF wavelength sigma
    mnlsfwsigma = np.nanmean(lsfwsigma,axis=0)
    # polynomial fit to the values versus wavelength
    gdw, = np.where(np.isfinite(mnlsfwsigma))
    lsfcoef = np.polyfit(fwave[gdw],mnlsfwsigma[gdw],2)
                
    # Create final spectrum
    zeros = np.zeros(nfpix)
    izeros = np.zeros(nfpix,bool)
    comb = Spec1D(zeros,err=zeros.copy(),mask=np.ones(nfpix,bool),wave=stack.wave.copy(),
                  instrument='NIRSpec',lsfpars=lsfcoef[::-1],lsftype=spec.lsf.lsftype,lsfxtype=spec.lsf.xtype)
    comb.cont = zeros.copy()

    # Pixel-by-pixel weighted average
    cont = np.mean(stack.cont,axis=0)
    comb.flux = np.sum(stack.flux/stack.err**2,axis=0)/np.sum(1./stack.err**2,axis=0) * cont
    comb.err =  np.sqrt(1./np.sum(1./stack.err**2,axis=0)) * cont
    comb.mask = np.bitwise_and.reduce(stack.mask,0)
    comb.err[comb.mask] = 1e30
    comb.cont = cont
    comb.bc = np.mean(stack.bc)
    if hasfluxcorr:
        comb.fluxcorr = np.mean(stack.fluxcorr,axis=0)
    
    return comb,stack

def process(fileinput,outdir='./',clobber=False):
    """ Process multiple rate files using JWST calwebb_spec2 pipeline."""
    files = glob(fileinput)
    nfiles = len(files)
    print(nfiles,' files found for ',fileinput)
    if nfiles==0:
        return
    for f in files: print(f)
    
    # File loop
    for i in range(nfiles):
        filename = files[i]
        print(' ')
        print('------------------------------------------------------------')        
        print(i+1,filename)
        print('------------------------------------------------------------')
        print(' ')
        # Check that it's a dispersed exposures
        hdu = fits.open(filename)
        filt = hdu[0].header['filter']
        grating = hdu[0].header['grating']
        hdu.close()
        if grating=='MIRROR':
            print('NOT a dispersed image')
            continue
        res = process_exp(filename,outdir=outdir,clobber=clobber)
    
    

def process_exp(filename,outdir='./',clobber=False):
    """ Process exposure image through the JWST calwebb_spec2 pipeline."""
    # filename should be a rate filename
    
    # Do NOT perform image-to-image background subtraction
    # Extend the slits to be at least 5 shutters long

    if os.path.exists(filename)==False:
        raise ValueError(filename+' NOT FOUND')
    if filename.endswith('_rate.fits')==False:
        raise ValueError('Must be rate file')

    odir = os.path.abspath(outdir)
    if os.path.exists(odir)==False:
        os.makedirs(odir)
    
    filebase = os.path.basename(filename)[:-10]  # remove _rate.fits
    print('Processing ',filename)

    outfile = odir+'/'+filebase+'_red.fits'
    if os.path.exists(outfile) and clobber==False:
        print(outfile+' exists already and clobber==False')
        return None
    
    # Do the processing in a temporary directory
    tempdir = tempfile.mkdtemp(prefix='msaproc',dir='./')
    tempdir = os.path.abspath(tempdir)
    
    # Make symlink to the rate file
    oldfilename = os.path.abspath(filename)
    newfilename = tempdir+'/'+os.path.basename(filename)
    os.symlink(oldfilename,newfilename)
    
    # Get some metadata
    hdu = fits.open(filename)
    msa_filename = hdu[0].header['MSAMETFL']
    msa_metadata_id = hdu[0].header['MSAMETID']
    dither_position = hdu[0].header['PATT_NUM']
    hdu.close()
    if os.path.exists(msa_filename)==False:
        msa_filename = os.path.dirname(oldfilename)+'/'+msa_filename
    if os.path.exists(msa_filename)==False:
        raise ValueError(msa_filename+' NOT FOUND')    

    # Expand MSA slits
    msahdu = fits.open(msa_filename)
    tab = Table(msahdu[2].data)
    newtab = utils.expand_msa_slits(tab,msa_metadata_id=msa_metadata_id,dither_position=dither_position)
        
    # Move to temporary directory
    curdir = os.path.abspath(os.curdir)
    os.chdir(tempdir)

    # Write the MSA file in the temporary directory
    newhdu = fits.HDUList()
    newhdu.append(msahdu[0])
    newhdu.append(msahdu[1])
    newhdu.append(fits.table_to_hdu(newtab))
    newhdu[2].header['EXTNAME'] = 'SHUTTER_INFO'
    newhdu.append(msahdu[3])
    newmsa_filename = tempdir+'/'+os.path.basename(msa_filename)
    newhdu.writeto(newmsa_filename,overwrite=True)
    newhdu.close()
    msahdu.close()
    
    # Create an instance of the pipeline class
    spec2 = Spec2Pipeline()
    # Set some parameters that pertain to the entire pipeline
    spec2.save_results = False    
    #spec2.assign_wcs.skip = True
    spec2.bkg_subtract.skip = True
    #spec2.imprint_subtract.skip = True
    #spec2.msa_flagging.skip = True
    #spec2.extract_2d.skip = True
    #spec2.srctype.skip = True
    #spec2.master_background_mos.skip = True
    #spec2.wavecorr.skip = True
    #spec2.flat_field.skip = True
    #spec2.pathloss.skip = True
    #spec2.barshadow.skip = True
    #spec2.photom.skip = True
    spec2.resample_spec.skip = True
    spec2.extract_1d.skip = True

    rate_file = os.path.basename(newfilename)
    result = spec2.run(rate_file)
    if type(result)==list:
        result = result[0]
        
    # Move the modified MSA file to the output directory
    outmsafile = odir+'/'+filebase+'_'+msa_filename[-11:]  # keep the msa number at the end
    print('Saving modified MSA file to ',outmsafile)
    shutil.move(newmsa_filename,outmsafile)    
    # Save results to output directory
    print('Saving results to ',outfile)
    result.save(outfile)
    
    # Delete the temporary directory
    os.chdir(curdir)
    shutil.rmtree(tempdir)

    return result


def reduce(obsname,outdir='./',logger=None,clobber=False,redtag='red',noback=False,fluxcorrfile=None):
    """ This extracts spectra from the JWST NIRSpec MSA data """

    #if fluxcorrfile is None:
    #    fluxcorrfile = '/Users/nidever/jwst/2609/nirspec/nirspec_fluxcorr.fits'
    #    # Load the fluxcorr file
    #    ffluxcorr = fits.getdata(fluxcorrfile)
    #    # wavelength coefficients with linear wavelength steps
    #    # 3834 pixels, from 9799.765 to 18797.7624 A
    #    wcoef = np.array([-1.35698061e-09, -7.79636391e-06,  2.39732634e+00,  9.79971041e+03])
    #    npix = 3834
    #    xpix = np.arange(npix)
    #    wfluxcorr = np.polyval(wcoef,xpix)
    #    fluxcorr = {'flux':ffluxcorr,'wave':wfluxcorr}
    fluxcorr = None
    
    if outdir.endswith('/')==False: outdir+='/'
    #if logger is None: logger=dln.basiclogger()
    
    # Get exposures information
    edict = getexpinfo(obsname,redtag=redtag)
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
        stackdir = odir+'stack/'
        if os.path.exists(stackdir)==False: os.makedirs(stackdir)
        stackplotdir = stackdir+'plots/'
        if os.path.exists(stackplotdir)==False:
            os.makedirs(stackplotdir)
            
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
        slexpspec = []
        expspec = []        
        sourcenames = []
        for i in range(nexp):
            expname = expnames[i]
            print(' ')
            print('------------------------------------')            
            print('EXPOSURE '+str(i+1)+' '+expname)
            print('------------------------------------')
            if nexp>1 and noback==False:
                if i==0:
                    backexpname = expnames[1]
                else:
                    backexpname = expnames[0]
            else:
                backexpname = None

            #print('KLUDGE!!!!')
            #expname = 'jw02609006001_03101_00003'
            speclist = extractexp(expname,backexpname,outdir=odir,redtag=redtag,clobber=clobber,fluxcorr=fluxcorr)
            srcname = [s.source_name for s in speclist]
            #slexpspec.append(slspeclist)
            expspec.append(speclist)            
            sourcenames += srcname

        #import pdb; pdb.set_trace()
            
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
                #slsplist = []
                splist = []                
                for e in range(nexp):
                    #eslspeclist = slexpspec[e]  # list of all spectra from this exposures
                    especlist = expspec[e]  # list of all spectra from this exposures                    
                    esourcename = np.array([s.source_name for s in especlist])
                    ind, = np.where(esourcename==srcname)
                    if len(ind)>0:
                        #slsplist.append(eslspeclist[ind[0]])
                        splist.append(especlist[ind[0]])                        
                        
                # Do the stacking
                if len(splist)>1:
                    print('Combining spectra from multiple exposures')
                    #combslsp,slstack = stackspec(slsplist)
                    combsp,stack = stackspec(splist)                    
                else:
                    #combslsp = slsplist[0]
                    combsp = splist[0]                    
            else:
                import pdb; pdb.set_trace()
                #combslsp = slsplist[0]
                combsp = splist[0]                
                
            # Write to file
            outfile = stackdir+'/spStack-'+srcname+'_'+redtag+'.fits'
            print('Writing to '+outfile)
            combsp.write(outfile,overwrite=True)
            #combslsp.write(outfile,overwrite=True)            
            #combsp.write(outfile.replace('_cal.fits','_rate.fits'),overwrite=True)            

            # Save a plot
            matplotlib.use('Agg')
            fig = plt.figure(figsize=(12,7))
            plt.clf()
            medflux = np.nanmedian(combsp.flux)
            plt.plot(combsp.wave,combsp.flux)
            plt.xlabel('X')
            plt.ylabel('Flux')
            plt.ylim(-medflux/3.,1.8*medflux)
            plt.savefig(stackplotdir+'/spStack-'+srcname+'_'+redtag+'_flux.png',bbox_inches='tight')
            plt.clf()
            #medflux = np.nanmedian(combsp.flux)            
            #plt.plot(combsp.wave,combsp.flux)
            #plt.xlabel('X')
            #plt.ylabel('Flux')
            #plt.ylim(-medflux/3.,1.8*medflux)            
            #plt.savefig(stackplotdir+'/spStack-'+srcname+'_rate_flux.png',bbox_inches='tight')
            matplotlib.use('MacOSX')


        # Run qa
        qa.qa(obsid)
            
    print('Done')



def extractexp(expname,backexpname=None,logger=None,outdir='./',clobber=False,redtag='red',fluxcorr=None):
    """ This performs 1D-extraction of the spectra in one exposure."""

    #if logger is None: logger=dln.basiclogger()
    plotdir = outdir+'/plots/'
    if os.path.exists(plotdir)==False:
        os.makedirs(plotdir)
        
    # Load the calibrated file    
    calfilename1 = expname+'_nrs1_'+redtag+'.fits'
    calfilename2 = expname+'_nrs2_'+redtag+'.fits'
    origcalfilename1 = expname+'_nrs1_cal.fits'
    origcalfilename2 = expname+'_nrs2_cal.fits'    
    #calfilename1 = expname+'_nrs1/'+expname+'_nrs1_'+redtag+'.fits'    
    #ratefilename1 = expname+'_nrs1/'+expname+'_nrs1_rate.fits'    
    #calfilename2 = expname+'_nrs2/'+expname+'_nrs2_'+redtag+'.fits'
    #ratefilename2 = expname+'_nrs2/'+expname+'_nrs2_rate.fits'
    #for f in [calfilename1,ratefilename1,calfilename1,ratefilename1]:
    for f in [calfilename1,calfilename1]:        
        if os.path.exists(f)==False:
            raise ValueError(f+' NOT FOUND')
    if backexpname is not None:
        #bratefilename1 = backexpname+'_nrs1/'+backexpname+'_nrs1_rate.fits'
        #bratefilename2 = backexpname+'_nrs2/'+backexpname+'_nrs2_rate.fits'
        #if os.path.exists(bratefilename1)==False:
        #    raise ValueError(bratefilename1+' NOT FOUND')
        #if os.path.exists(bratefilename2)==False:
        #    raise ValueError(bratefilename2+' NOT FOUND')        
        bcalfilename1 = backexpname+'_nrs1_'+redtag+'.fits'
        bcalfilename2 = backexpname+'_nrs2_'+redtag+'.fits'        
        if os.path.exists(bcalfilename1)==False:
            raise ValueError(bcalfilename1+' NOT FOUND')
        if os.path.exists(bcalfilename2)==False:
            raise ValueError(bcalfilename2+' NOT FOUND')        
        
    hdu1 = fits.open(calfilename1)
    nsources1 = int(len(hdu1)-2/8)
    sourceid1 = []
    sourcename1 = []    
    for i in np.arange(1,len(hdu1)):
        if hdu1[i].header.get('extname')=='SCI':
            sourceid1.append(hdu1[i].header['srcalias'])
            sourcename1.append(hdu1[i].header['srcname']) 
    sourceid1 = np.array(sourceid1)
    hdu1.close()
    hdu2 = fits.open(calfilename2)
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

    # Get sourceids for the background exposures
    if backexpname is not None:
        bhdu1 = fits.open(bcalfilename1)
        nbsources1 = int(len(bhdu1)-2/8)
        bsourceid1 = []
        bsourcename1 = []    
        for i in np.arange(1,len(bhdu1)):
            if bhdu1[i].header.get('extname')=='SCI':
                bsourceid1.append(bhdu1[i].header['srcalias'])
                bsourcename1.append(bhdu1[i].header['srcname']) 
        bsourceid1 = np.array(bsourceid1)
        bhdu1.close()
        bhdu2 = fits.open(bcalfilename2)
        bsourceid2 = []
        bsourcename2 = []
        for i in np.arange(1,len(bhdu2)):
            if bhdu2[i].header.get('extname')=='SCI':
                bsourceid2.append(bhdu2[i].header['srcalias'])
                bsourcename2.append(bhdu2[i].header['srcname'])
        bsourceid2 = np.array(bsourceid2)
        bhdu2.close()

    
    # Looping over sources
    data1,data2,rate1,rate2,brate1,brate2 = None,None,None,None,None,None
    ocalhdu1,ocalhdu2 = None,None
    backdata1,backdata2 = None,None
    slspeclist = []
    speclist = []    
    for i in range(nsources):
        sourceid = sourceids[i]
        sourcename = sourcenames[i]
        sp = None
        print(' ')
        print('--',i+1,sourceid,'--')
        
        # Check if it already exists
        outfile = outdir+'spVisit-'+sourcename+'_'+expname+'_'+redtag+'.fits'
        if os.path.exists(outfile) and clobber==False:
            if os.path.getsize(outfile)==0:
                print(outfile+' is an empty file.')
                continue
            print(outfile+' already exists. Loading')            
            #slsp = doppler.read(outfile)
            #slspeclist.append(slsp)
            #sp = doppler.read(outfile.replace('_cal.fits','_rate.fits'))
            sp = doppler.read(outfile)
            speclist.append(sp)
            continue

        if data1 is None:
            print('Loading '+calfilename1)
            data1 = datamodels.open(calfilename1)
        if data2 is None:
            print('Loading '+calfilename2)    
            data2 = datamodels.open(calfilename2)
        #if ocalhdu1 is None:
        #    ocalhdu1 = fits.open(origcalfilename1)
        #if ocalhdu2 is None:
        #    ocalhdu1 = fits.open(origcalfilename2)            
        #if rate1 is None:
        #    rate1 = fits.open(ratefilename1)
        #if rate2 is None:
        #    rate2 = fits.open(ratefilename2)            
        if backexpname is not None:
            #if brate1 is None:
            #    brate1 = fits.open(bratefilename1)
            #if brate2 is None:
            #    brate2 = fits.open(bratefilename2)
            if backdata1 is None:
                print('Loading '+bcalfilename1)
                backdata1 = datamodels.open(bcalfilename1)
            if backdata2 is None:
                print('Loading '+bcalfilename2)                
                backdata2 = datamodels.open(bcalfilename2)            
                
        # NRS1
        print('-NRS1-')
        slsp1,sp1,backslit1,bind1 = None,None,None,[]
        ind1, = np.where(sourceid1==sourceid)
        if backexpname is not None:
            bind1, = np.where(bsourceid1==sourceid)
        if len(ind1)>0:
            plotbase = plotdir+sourcename+'_'+expname+'_nrs1'
            #slsp1,sp1 = extract.extract_slit(data1,data1.slits[ind1[0]],rate1,brate1,plotbase=plotbase)
            if len(bind1)>0:
                backslit1 = backdata1.slits[bind1[0]]
            #sp1 = extract.extract_slit(data1,data1.slits[ind1[0]],backslit1,ocalhdu1[ind1[0]*10+1],plotbase=plotbase)
            try:
                sp1 = extract.extract_slit(data1,data1.slits[ind1[0]],backslit1,applyslitcorr=True,plotbase=plotbase)
                # Apply the flux correction
                if sp1 is not None and fluxcorr is not None:
                    fluxcorr2 = dln.interp(fluxcorr['wave'],fluxcorr['flux'],sp1.wave)
                    sp1.flux /= fluxcorr2
                    sp1.err /= fluxcorr2                    
                    sp1.fluxcorr = fluxcorr2                
            except:
                traceback.print_exc()
        # NRS2
        print('-NRS2-')        
        slsp2,sp2,backslit2,bind2 = None,None,None,[]
        ind2, = np.where(sourceid2==sourceid)
        if backexpname is not None:
            bind2, = np.where(bsourceid2==sourceid)        
        if len(ind2)>0:
            plotbase = plotdir+sourcename+'_'+expname+'_nrs2'
            #slsp2,sp2 = extract.extract_slit(data2,data2.slits[ind2[0]],rate2,brate2,plotbase=plotbase)
            if len(bind2)>0:
                backslit2 = backdata2.slits[bind2[0]]
            #sp2 = extract.extract_slit(data2,data2.slits[ind2[0]],backslit2,ocalhdu2[ind2[0]*10+1],plotbase=plotbase)
            try:
                sp2 = extract.extract_slit(data2,data2.slits[ind2[0]],backslit2,applyslitcorr=True,plotbase=plotbase)
                # Apply the flux correction
                if sp2 is not None and fluxcorr is not None:
                    fluxcorr2 = dln.interp(fluxcorr['wave'],fluxcorr['flux'],sp2.wave)
                    sp2.flux /= fluxcorr2
                    sp2.err /= fluxcorr2                    
                    sp2.fluxcorr = fluxcorr2
            except:
                traceback.print_exc()
                
        # Join the two spectra together
        if sp1 is not None and sp2 is not None:
            #slsp = joinspec(slsp1,slsp2)
            sp = joinspec(sp1,sp2)            
        else:
            if sp1 is not None:
                #slsp = slsp1
                sp = sp1                
            if sp2 is not None:
                #slsp = slsp2
                sp = sp2                
                
        # Save the file
        if sp is not None:
            print('Writing to '+outfile)
            sp.write(outfile,overwrite=True)
            #slsp.write(outfile,overwrite=True)            
            #sp.write(outfile.replace('_cal.fits','_rate.fits'),overwrite=True)            
            #slspeclist.append(slsp)
            speclist.append(sp)            
        else:
            dln.touch(outfile)
            
    # Close the files
    if data1 is not None: data1.close()
    if data2 is not None: data2.close()
    if ocalhdu1 is not None: ocalhdu1.close()
    if ocalhdu2 is not None: ocalhdu2.close()    
    if backdata1 is not None: backdata1.close()
    if backdata2 is not None: backdata2.close()    
    #if rate1 is not None: rate1.close()
    #if rate2 is not None: rate2.close()
    #if brate1 is not None: brate1.close()
    #if brate2 is not None: brate2.close()        
    
    #return slspeclist,speclist
    return speclist
