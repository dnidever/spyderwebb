#!/usr/bin/env python

"""EXTINCTION.PY - Add extinction to isochrones

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210919'  # yyyymmdd

import os
import numpy as np
from glob import glob
from astropy.io import fits
from astropy.table import Table
from . import utils


def load(filename=None):
    """ Load extinctions."""
    ddir = utils.datadir()
    if filename is None:
        files = glob(ddir+'extinctions.txt')
        nfiles = len(files)
        if nfiles==0:
            raise Exception("No default extinctions file found in "+ddir)
        filename = files[0]
    else:
        if os.path.exists(filename)==False:
            raise Exception(filename+' NOT FOUND')
    tab = Table.read(filename,format='ascii')
    # Turn into dictionary
    ext = {}
    for i in range(len(tab)):
        ext[tab['NAME'][i]] = tab['EXTINCTION'][i]
    return ext,tab
        
def extinct(iso,ext,isonames=None,extdict=None,dataonly=None,verbose=False):
    """ Apply extinction to the photometry."""

    if verbose:
        print('Applying extinction of '+str(ext))

    # Table or Isochrone object
    if hasattr(iso,'data'):
        data = iso.data
    else:
        data = iso
        
    # Load the extinction
    if extdict is None:
        extdict = load()
        
    # names
    if isonames is None:
        if hasattr(iso,'bands'):
            isonames = iso.bands
        else:
            raise ValueError('isonames required')
            
    # photometry data
    phot = []
    for n in isonames:
        mag = data[n]
        magext = extdict[n]*ext  # add extinction
        if dataonly:
            phot.append(mag+magext)
        else:
            data[n] += magext

    if dataonly:
        phot = np.vstack(tuple(phot)).T
        return phot
    else:
        # Isochrone object, stuff data back in
        if hasattr(iso,'data'):
            iso.data = data
            return iso
        else:
            return data
