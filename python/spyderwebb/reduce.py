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
from jwst.extract_1d import jextract

from doppler.spec1d import Spec1D
from . import extract

def reduce(files):
    """ This reduces the JWST NIRSpec MSA data """
    pass


def extract1d(filename):
    """ This performs 1D-extraction of the spectra."""

    # Load the calibrated file
    if os.path.exists(filename)==False:
        raise ValueError(filename+' NOT FOUND')
    print('Loading '+filename)
    input_model = datamodels.open(filename)
    print('data is '+str(type(input_model)))
    nslits = len(input_model.slits)
    print(str(nslits)+' slits')
    filebase,ext = os.path.splitext(os.path.basename(filename))
    
    # Get the reference file
    print('Getting reference file')
    step = Extract1dStep()
    extract_ref = step.get_reference_file(input_model,'extract1d')
    print('Using reference file: '+extract_ref)
    extract_ref_dict = jextract.open_extract1d_ref(extract_ref, input_model.meta.exposure.type)

        
    # Looping over slits
    for i in range(nslits):
        slit = input_model.slits[i]
        print('Slit ',i+1)
        print('source_name:',slit.source_name)
        print('source_id:',slit.source_id)
        print('slitlet_id:',slit.slitlet_id)
        print('source ra/dec:',slit.source_ra,slit.source_dec)
        
        # Get the data
        im = slit.data
        err = slit.err
        wave = slit.wavelength
        ny,nx = im.shape

        # Get extraction parameters, from extract.create_extraction()
        slitname = slit.name
        sp_order = jextract.get_spectral_order(slit)        
        meta_source = slit
        smoothing_length = None
        use_source_posn = True
        extract_params = jextract.get_extract_parameters(extract_ref_dict,meta_source,slitname,
                                                         sp_order,input_model.meta,step.smoothing_length,
                                                         step.bkg_fit,step.bkg_order,use_source_posn)
        extract_params['dispaxis'] = extract_ref_dict['apertures'][0]['dispaxis']

        # Get extraction model, extract.extract_one_slit()
        # If there is an extract1d reference file (there doesn't have to be), it's in JSON format.
        extract_model = jextract.ExtractModel(input_model=input_model, slit=slit, **extract_params)
        ap = jextract.get_aperture(im.shape, extract_model.wcs, extract_params)
        jextract_model.update_extraction_limits(ap)
        
        if extract_model.use_source_posn:
            #if prev_offset == OFFSET_NOT_ASSIGNED_YET:  # Only call this method for the first integration.
            offset, locn = extract_model.offset_from_offset(input_model, slit)

            #if offset is not None and locn is not None:
            #    log.debug(f"Computed source offset={offset:.2f}, source location={locn:.2f}")

            if not extract_model.use_source_posn:
                offset = 0.
            #else:
            #    offset = prev_offset
        else:
            offset = 0.

        extract_model.position_correction = offset

        # Add the source position offset to the polynomial coefficients, or shift the reference image
        # (depending on the type of reference file).
        extract_model.add_position_correction(im.shape)
        extract_model.log_extraction_parameters()
        extract_model.assign_polynomial_limits()

        # get disp_range, from extract.ExtractModel.extract()
        if extract_model.dispaxis == extract.HORIZONTAL:
            slice0 = int(round(extract_model.xstart))
            slice1 = int(round(extract_model.xstop)) + 1
            #x_array = np.arange(slice0, slice1, dtype=np.float64)
            #y_array = np.empty(x_array.shape, dtype=np.float64)
            #y_array.fill((extract_model.ystart + extract_model.ystop) / 2.)
        else:
            slice0 = int(round(extract_model.ystart))
            slice1 = int(round(extract_model.ystop)) + 1
            #y_array = np.arange(slice0, slice1, dtype=np.float64)
            #x_array = np.empty(y_array.shape, dtype=np.float64)
            #x_array.fill((extract_model.xstart + extract_model.xstop) / 2.)
        disp_range = [slice0, slice1]  # Range (slice) of pixel numbers in the dispersion direction.
          
        # Get the trace, from extract1d.extract1d()
        p_src = extract_model.p_src
        srclim = []                 # this will be a list of lists, like p_src
        n_srclim = len(p_src)

        for i in range(n_srclim):
            lower = p_src[i][0]
            upper = p_src[i][1]
            if extract_model.independent_var.startswith("wavelength"):    # OK if 'wavelengths'
                srclim.append([lower(lambdas), upper(lambdas)])
            else:
                # Temporary array for the independent variable.
                pixels = np.arange(disp_range[0], disp_range[1], dtype=np.float64)
                srclim.append([lower(pixels), upper(pixels)])


        # Polynomial coefficients for traces
        # y(1024) = 0
        coef0 = np.array([ 4.59440169e-06, -1.86120908e-04, -4.83294422e+00])

        # Get peak at X=1024
        tot = np.nansum(np.maximum(im[:,974:1075],0),axis=1)
        yind = np.argmax(tot)
        # Trace for this star
        coef = coef0.copy()
        coef[2] += yind
        x = np.arange(2048)
        ytrace = np.polyval(coef,x)

        # Get the trace
        ttab = extract.tracing(im,yind)
        tcoef = robust.polyfit(ttab['x'],ttab['y'],2)
        ytrace = np.polyval(tcoef,x)
        tsigcoef = robust.polyfit(ttab['x'],ttab['ysig'],1)
        ysig = np.polyval(tsigcoef,x)
        
        # Create the mask
        ybin = 3
        yy = np.arange(ny).reshape(-1,1) + np.zeros(nx).reshape(1,-1)
        mask = ((yy >= (ytrace-ybin)) & (yy <= (ytrace+ybin)))

        # I think the trace is at
        # ny/2+offset at Y=1024

        
        # Build PSF model

        # Boxcar
        boxsum = np.nansum(mask*im,axis=0)

        # Optimal extraction
        flux,fluxerr,trace = extract.extract_optimal(im*mask,ytrace,imerr=err*mask,verbose=verbose,
                                                     off=10,backoff=50,smlen=31)
        # Get the wavelengths
        wav = np.nansum(wave*mask,axis=0)/np.sum(mask,axis=0) * 1e4  # convert to Angstroms
        
        # Apply slit correction
        srcxpos = slit.source_xpos
        srcypos = slit.source_ypos
        # SLIT correction, srcxpos is source position in slit
        # the slit is 2 pixels wide
        dwave = np.gradient(wav)
        newwav = wav+2*srcxpos*dwave
        print('Applying slit correction: %.2f pixels' % (2*srcxpos))
        

        # Apply relative flux calibration correction

        # Put it all together
        sp = Spec1D(flux,err=fluxerr,wave=wav,instrument='NIRSpec')
        sp.ytrace = trace
        sp.source_name = slit.source_name
        sp.source_id = slit.source_id
        sp.slitlet_id = slit.slitlet_id
        sp.source_ra = slit.source_ra
        sp.source_dec = slit.source_dec        
        #spec = Table((newwav,flux,fluxerr,trace),names=['wave','flux','flux_error','ytrace'])

        # Save the file
        filename = slit.source_name+'_'+filebase+'.fits'
        print('Writing spectrum to ',filename)
        sp.write(filename,overwrite=True)
        
    import pdb; pdb.set_trace()
