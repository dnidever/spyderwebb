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

from glob import glob
from doppler.spec1d import Spec1D
from . import extract

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
    
def reduce(obsname,outdir='./',logger=None):
    """ This reduces the JWST NIRSpec MSA data """

    if outdir.endswith('/')==False: outdir+='/'
    #if logger is None: logger=dln.basiclogger()

    
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
        sourceids = []
        for i in range(nexp):
            expname = expnames[i]
            print(' ')
            print('------------------------------------')            
            print('EXPOSURE '+str(i+1)+' '+expname)
            print('------------------------------------')
            speclist = extractexp(expname,outdir=odir)
            srcid = [s.source_id for s in speclist]
            expspec.append(speclist)
            sourceids += srcid

        # Stack spectra from multiple exposures
        if nexp>1:
            print('Combining spectra from multiple exposures')
            # Loop over sources
            sourceid = np.unique(np.array(sourceids))
            nsources = len(sourceid)
            for i in range(nsources):
                srcid = sourceid[i]
                # Loop over exposures
                splist = []
                for e in rang(nexp):
                    especlist = expspec[e]  # list of all spectra from this exposures
                    esourceid = np.array([s.source_id for s in especlist])
                    ind, = np.where(esourceid==srcid)
                    if len(ind)>0:
                        splist.append(especlist[ind[0]])

                # Do the stacking

                # Write to file
                
                import pdb; pdb.set_trace()
            
        import pdb; pdb.set_trace()



def extractexp(expname,logger=None,outdir='./'):
    """ This performs 1D-extraction of the spectra in one exposure."""

    #if logger is None: logger=dln.basiclogger()

    # Load the calibrated file    
    filename1 = expname+'_nrs1/'+expname+'_nrs1_cal.fits'
    filename2 = expname+'_nrs2/'+expname+'_nrs2_cal.fits'
    if os.path.exists(filename1)==False:
        raise ValueError(filename1+' NOT FOUND')
    if os.path.exists(filename2)==False:
        raise ValueError(filename2+' NOT FOUND')
    print('Loading '+filename1)
    data1 = datamodels.open(filename1)
    print('Loading '+filename2)    
    data2 = datamodels.open(filename2)

    # Get source_ids
    sourceid1 = np.array([s.source_id for s in data1.slits])
    sourceid2 = np.array([s.source_id for s in data2.slits])    
    sourceids = np.unique(np.hstack((sourceid1,sourceid2)))
    nsources = len(sourceids)
    print(str(nsources)+' sources')

    # Looping over sources
    speclist = []
    for i in range(nsources):
        sourceid = sourceids[i]
        print(i+1,sourceid)
        # NRS1
        ind1, = np.where(sourceid1==sourceid)
        if len(ind1)>0:
            sp1 = extract.extract_slit(data1,data1.slits[ind1[0]])
        # NRS2
        ind2, = np.where(sourceid2==sourceid)
        if len(ind2)>0:
            sp2 = extract.extract_slit(data2,data2.slits[ind2[0]])

        # Join the two spectra together
        sp = joinspec(sp1,sp2)

        import pdb;pdb.set_trace()
        
        # Save the file
        outfile = outdir+sp.source_name+'_'+expname+'.fits'
        print('Writing to '+outfile)
        sp.write(outfile,overwrite=True)
        
        speclist.append(sp)

    # Close the files
    data1.close()
    data2.close()
        
    return speclist
