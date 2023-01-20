import os
import numpy as np
from glob import glob
from dlnpyutils import utils as dln
import doppler
from doppler import spec1d

def continuum(sp,order=1):
    return spec1d.continuum(sp,norder=2,perclevel=50.0,binsize=0.2,interp=True)
    #x = np.linspace(-1,1,len(sp.wave))
    #good1 = (~sp.mask)
    #coef1,_ = dln.ladfit(x[good1],sp.flux[good1])
    #coef1 = coef1[::-1]
    #cont1 = np.polyval(coef1,x)    
    #sig = dln.mad(sp.flux/cont1)
    #good = ((np.abs(sp.flux/cont1-1) < 3*sig) & ~sp.mask)
    #if order==1:
    #    coef,_ = dln.ladfit(x[good],sp.flux[good])
    #    coef = coef[::-1]
    #else:   
    #    coef = robust.polyfit(x[good],sp.flux[good],order)
    #cont = np.polyval(coef,x)
    #return cont,coef

def readspec(filename):
    spec = doppler.read(filename)
    # Doppler models only go up to 18000
    bad = (spec.wave >= 18000)
    if np.sum(bad)>0:
        # Trim to wave < 18000
        gd, = np.where(spec.wave < 18000)
        orig = spec.copy()
        spec.flux = spec.flux[gd]
        spec.err = spec.err[gd]
        spec.wave = spec.wave[gd]
        spec.mask = spec.mask[gd]
        if hasattr(spec,'_cont'):
            spec._cont = spec._cont[gd]
    spec._cont = continuum(spec)
    return spec


def doppler_visit(visitfiles,payne=False,verbose=True):
    """ Run Doppler jointly on visit files for a star."""

    # Load the files
    spec = []
    for i in range(len(visitfiles)):
        sp = readspec(visitfiles[i])
        spec.append(sp)
    out = doppler.jointfit(spec,verbose=verbose,payne=payne)
    return out


def doppler_stack(stackfile,payne=False,verbose=True):
    """ Run Doppler on the stacked stellar spectrum."""
    spec = readspec(stackfile)
    out = doppler.fit(spec,verbose=verbose,payne=payne)
    return out

def run_doppler(obsid,redtag='red'):
    """ Run Doppler on all the stars."""
    # obsid 'G140H-F100LP-M71_test68'
    
    print('Running doppler for '+obsid)
    
    # Get visit and stack files
    visitfiles = np.char.array(glob(obsid+'/spVisit-*'+redtag+'.fits'))
    visitbases = [os.path.basename(v) for v in visitfiles if os.path.getsize(v)>0]
    vstarids = [v[8:] for v in visitbases]
    vstarids = [v[:v.find('_jw')] for v in vstarids]
    vstarids = np.char.array(vstarids)
    nvisits = len(visitfiles)

    # Get the list of Stacked files    
    stackfiles = np.char.array(glob(obsid+'/stack/spStack-*_'+redtag+'.fits*'))
    starids = [os.path.basename(c)[8:-len(redtag)-6] for c in stackfiles if os.path.getsize(c)>0]
    starids = np.char.array(starids)
    nstars = len(starids)
    print(nstars,' stars')

    # Loop over the stars
    for i in range(nstars):
        starid = starids[i]
        print(i+1,starid)
        # Visit files
        vind, = np.where(vstarids==starid)
        vfiles = visitfiles[vind]
        #vout = doppler_visit(vfiles)
        
        # Stack file
        sind, = np.where(starids==starid)
        sfiles = stackfiles[sind[0]]
        sout = doppler_stack(sfiles)

    import pdb; pdb.set_trace()
