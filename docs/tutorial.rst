********
Tutorial
********


Running SPyderWebb
==================
In order to be able to run |spyderwebb| you will need to install the JWST calibration pipeline.  It is best to create a python environment for this.

.. code-block:: python

    # m71/
    from spyderwebb import reduc
    reduce.process('jw02609006001_03101_0000?_nrs?/*_rate.fits',outdir='red')

    # m71/red/
    obsname = 'jw02609006001_03101'
    out=reduce.reduce(obsname,clobber=True,noback=True)

		
This will create an output file called ``spectrum_doppler.fits``.

