********
Tutorial
********


Running SPyderWebb
==================
In order to be able to run |spyderwebb| you will need to install the JWST calibration pipeline.  It is best to create a python environment for this.

Step 1: Download Data
---------------------
Download your data from `MAST <https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html>`_.
You want all of your ``_rate.fits`` files and the ``_msa.fits`` file which contains the information about
what shutters the sources are in.

Each exposure and detector should have its own directory.  That's what |spyderwebb| expects. For example:

.. code-block:: bash

    jw02609006001_03101_00002_nrs1/
       jw02609006001_01_msa.fits
       jw02609006001_03101_00002_nrs1_rate.fits
    jw02609006001_03101_00002_nrs2/
       jw02609006001_03101_00002_nrs2_rate.fits
    jw02609006001_03101_00003_nrs1/
       jw02609006001_03101_00003_nrs1_rate.fits
    jw02609006001_03101_00003_nrs2/
       jw02609006001_03101_00003_nrs2_rate.fits


.. code-block:: python

    # m71/
    from spyderwebb import reduc
    reduce.process('jw02609006001_03101_0000?_nrs?/*_rate.fits',outdir='red')


    
    # m71/red/
    obsname = 'jw02609006001_03101'
    out=reduce.reduce(obsname,clobber=True,noback=True)

		
This will create an output file called ``spectrum_doppler.fits``.

