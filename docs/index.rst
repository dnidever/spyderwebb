.. spyderwebb documentation master file, created by
   sphinx-quickstart on Tue Feb 16 13:03:42 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**********
SPyderWebb
**********

Introduction
============
|SPyderWebb| is Python software package to reduce JWST NIRSPec MSA data.  It is built on the `JWST Calibration ppeline<https://jwst-docs.stsci.edu/jwst-science-calibration-pipeline>` but performs certain tasks differently.

.. toctree::
   :maxdepth: 1

   install
   gettingstarted
   examples
   modules
	      

Description
===========
|SPyderWebb| reduces and extracts JWST NIRSpec spectra.

The main steps are:

1. Run the main JWST calibration pipeline with some modifications
   - Do not perform the normal background subtraction.  The normal method is to take a dithered exposure and subtract it from the main exposure.  This just adds noise and increases the noise by sqrt(2).  Performing background subtraction during the extraction process is much better.
   - Increase the region that is extraction in the cal file.  By default, only the pixels of the shutter are "extracted" into the 2D cal image.  This is later used do the extraction to 1D and we need more pixels to perform better extraction and estimating the background.  In order to achieve this, the msa file has to be temporarily modified for each slit.
2. Extraction to 1D
   - optimal extraction
   - "fix" outlier pixels
   - slit correction
3. Run Doppler to determine radial velocities
4. Run FERRE with a 4D grid (Teff, logg, [M/H], [alpha/M]) to determine abundances and stellar parameters.

Examples
========

.. toctree::
    :maxdepth: 1

    examples
    gettingstarted

    # m71/
    from spyderwebb import reduc
    reduce.process('jw02609006001_03101_0000?_nrs?/*_rate.fits',outdir='red')

    # m71/red/
    obsname = 'jw02609006001_03101'
    out=reduce.reduce(obsname,clobber=True,noback=True)


*****
Index
*****

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`			  
			  
.. rubric:: Footnotes

.. [#f1] For `Christian Doppler <https://en.wikipedia.org/wiki/Christian_Doppler>`_ who was an Austrian physicist who discovered the `Doppler effect <https://en.wikipedia.org/wiki/Doppler_effect>`_ which is the change in frequency of a wave due to the relative speed of the source and observer.
