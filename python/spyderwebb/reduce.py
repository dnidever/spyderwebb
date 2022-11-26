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
    print('Found '+str(len(uobsids))+' observation group(s)')
    print(', '.join(uobsids))

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
        print('Processing observation group: '+obsid)
        print(str(nind)+' exposures')
        print('Using output directory: '+odir)

        # Loop over exposures and extract the spectra
        for i in range(nexp):
            expname = expnames[i]
            speclist = extractexp(expname)  

            import pdb; pdb.set_trace()
            
        import pdb; pdb.set_trace()


def extractexp(expname,logger=None):
    """ This performs 1D-extraction of the spectra."""

    #if logger is None: logger=dln.basiclogger()

    
    
    # Load the calibrated file    
    filename1 = expname+'_nrs1/'+expname+'_nrs1_cal.fits'
    filename2 = expname+'_nrs2/'+expname+'_nrs2_cal.fits'
    if os.path.exists(filename1)==False:
        raise ValueError(filename1+' NOT FOUND')
    if os.path.exists(filename2)==False:
        raise ValueError(filename2+' NOT FOUND')    
    data1 = datamodels.open(filename1)
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
        # NRS1
        ind1, = np.where(sourceid1==sourceid)
        if len(ind1)>0:
            sp1 = extract.extract_slit(data1.slits[ind1[0]])
        # NRS2
        ind2, = np.where(sourceid2==sourceid)
        if len(ind2)>0:
            sp2 = extract.extract_slit(data2.slits[ind2[0]])
        # Combine the two spectra

        import pdb;pdb.set_trace()
        
            
        speclist.append(sp)

    return speclist
