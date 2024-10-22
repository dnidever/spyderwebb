import os
import numpy as np
from gwcs.wcs import WCS
import jwst
from stdatamodels.jwst import datamodels
from jwst.lib.wcs_utils import get_wavelengths

from gwcs import coordinate_frames as cf
from gwcs.wcstools import grid_from_bounding_box
from stdatamodels.jwst.datamodels import (CollimatorModel, CameraModel, DisperserModel, FOREModel,
                                          IFUFOREModel, MSAModel, OTEModel, IFUPostModel, IFUSlicerModel,
                                          WavelengthrangeModel, FPAModel)
from stdatamodels.jwst.transforms.models import (Rotation3DToGWA, DirCos2Unitless, Slit2Msa,
                                                 AngleFromGratingEquation, WavelengthFromGratingEquation,
                                                 Gwa2Slit, Unitless2DirCos, Logical, Slit, Snell,
                                                 RefractionIndexFromPrism)

# JWST data models
from jwst import datamodels
from jwst.assign_wcs import pointing,nirspec

# Generate JWST NIRSpec MSA masks

def detector_to_gwa(reference_files, detector, disperser):
    """
    Transform from ``sca`` frame to ``gwa`` frame.

    Parameters
    ----------
    reference_files: dict
        Dictionary with reference files returned by CRDS.
    detector : str
        The detector keyword.
    disperser : dict
        A corrected disperser ASDF object.

    Returns
    -------
    model : `~astropy.modeling.core.Model` model.
        Transform from DETECTOR frame to GWA frame.

    """
    with FPAModel(reference_files['fpa']) as f:
        fpa = getattr(f, detector.lower() + '_model')

    # f.nrs2_model exists but is NOT a proper model, not sure why
        
    with CameraModel(reference_files['camera']) as f:
        camera = f.model

    angles = [disperser['theta_x'], disperser['theta_y'],
              disperser['theta_z'], disperser['tilt_y']]
    rotation = Rotation3DToGWA(angles, axes_order="xyzy", name='rotation')
    u2dircos = Unitless2DirCos(name='unitless2directional_cosines')
    # NIRSPEC 1- vs 0- based pixel coordinates issue #1781
    '''
    The pipeline works with 0-based pixel coordinates. The Nirspec model,
    stored in reference files, is also 0-based. However, the algorithm specified
    by the IDT team specifies that pixel coordinates are 1-based. This is
    implemented below as a Shift(-1) & Shift(-1) transform. This makes the Nirspec
    instrument WCS pipeline "special" as it requires 1-based inputs.
    As a consequence many steps have to be modified to provide 1-based coordinates
    to the WCS call if the instrument is Nirspec. This is not always easy, especially
    when the step has no knowledge of the instrument.
    This is the reason the algorithm is modified to accept 0-based coordinates.
    This will be discussed in the future with the INS and IDT teams and may be solved
    by changing the algorithm but for now

    model = (models.Shift(-1) & models.Shift(-1) | fpa | camera | u2dircos | rotation)

    is changed to

    model = models.Shift(1) & models.Shift(1) | \
            models.Shift(-1) & models.Shift(-1) | fpa | camera | u2dircos | rotation
    '''
    import pdb; pdb.set_trace()    
    
    model = fpa | camera | u2dircos | rotation
    return model
    

def validate_open_slits(input_model, open_slits, reference_files):
    """
    Remove slits which do not project on the detector from the list of open slits.
    For each slit computes the transform from the slit to the detector and
    determines the bounding box.

    Parameters
    ----------
    input_model : jwst.datamodels.JwstDataModel
        Input data model

    Returns
    -------
    slit2det : dict
        A dictionary with the slit to detector transform for each slit,
        {slit_id: astropy.modeling.Model}
    """

    def _is_valid_slit(domain):
        xlow, xhigh = domain[0]
        ylow, yhigh = domain[1]
        if (xlow >= 2048 or ylow >= 2048 or
                xhigh <= 0 or yhigh <= 0 or
                xhigh - xlow < 2 or yhigh - ylow < 1):
            return False
        else:
            return True

    det2dms = nirspec.dms_to_sca(input_model).inverse
    # read models from reference file
    disperser = DisperserModel(reference_files['disperser'])
    disperser = nirspec.correct_tilt(disperser, input_model.meta.instrument.gwa_xtilt,
                                     input_model.meta.instrument.gwa_ytilt)

    order, wrange = nirspec.get_spectral_order_wrange(input_model,
                                              reference_files['wavelengthrange'])

    input_model.meta.wcsinfo.waverange_start = wrange[0]
    input_model.meta.wcsinfo.waverange_end = wrange[1]
    input_model.meta.wcsinfo.spectral_order = order
    disperser.groovedensity = disperser.groove_density
    agreq = nirspec.angle_from_disperser(disperser, input_model)
    # GWA to detector
    det2gwa = detector_to_gwa(reference_files,
                              input_model.meta.instrument.detector,
                              disperser)
    gwa2det = det2gwa.inverse
    # collimator to GWA
    collimator2gwa = nirspec.collimator_to_gwa(reference_files, disperser)
    
    col2det = collimator2gwa & Identity(1) | Mapping((3, 0, 1, 2)) | agreq | \
        gwa2det | det2dms

    slit2msa = nirspec.slit_to_msa(open_slits, reference_files['msa'])

    for slit in slit2msa.slits:
        msa_transform = slit2msa.get_model(slit.name)
        msa2det = msa_transform & Identity(1) | col2det

        bb = nirspec.compute_bounding_box(msa2det, wrange, slit.ymin, slit.ymax)

        valid = _is_valid_slit(bb)
        if not valid:
            log.info("Removing slit {0} from the list of open slits because the "
                     "WCS bounding_box is completely outside the detector.".format(slit.name))
            idx = np.nonzero([s.name == slit.name for s in open_slits])[0][0]
            open_slits.pop(idx)

    return open_slits

def get_open_slits(input_model, reference_files=None, slit_y_range=[-.55, .55]):
    """Return the opened slits/shutters in a MOS or Fixed Slits exposure.
    """
    exp_type = input_model.meta.exposure.type.lower()
    lamp_mode = input_model.meta.instrument.lamp_mode
    if isinstance(lamp_mode, str):
        lamp_mode = lamp_mode.lower()
    else:
        lamp_mode = 'none'
    if exp_type in ["nrs_msaspec", "nrs_autoflat"] or ((exp_type in ["nrs_lamp", "nrs_autowave"]) and
                                                       (lamp_mode == "msaspec")):
        msa_metadata_file, msa_metadata_id, dither_point = nirspec.get_msa_metadata(
            input_model, reference_files)
        slits = nirspec.get_open_msa_slits(msa_metadata_file, msa_metadata_id, dither_point, slit_y_range)
    elif exp_type == "nrs_fixedslit":
        slits = nirspec.get_open_fixed_slits(input_model, slit_y_range)
    elif exp_type == "nrs_brightobj":
        slits = [Slit('S1600A1', 3, 0, 0, 0, slit_y_range[0], slit_y_range[1], 5, 1)]
    elif exp_type in ["nrs_lamp", "nrs_autowave"]:
        if lamp_mode in ['fixedslit', 'brightobj']:
            slits = nirspec.get_open_fixed_slits(input_model, slit_y_range)
    else:
        raise ValueError("EXP_TYPE {0} is not supported".format(exp_type.upper()))

    if reference_files is not None and slits:
        slits = validate_open_slits(input_model, slits, reference_files)
        log.info("Slits projected on detector {0}: {1}".format(input_model.meta.instrument.detector,
                                                               [sl.name for sl in slits]))
    if not slits:
        log_message = "No open slits fall on detector {0}.".format(input_model.meta.instrument.detector)
        log.critical(log_message)
        raise NoDataOnDetectorError(log_message)
    return slits

def slits_wcs(input_model, reference_files, slit_y_range):
    """
    The WCS pipeline for MOS and fixed slits.

    The coordinate frames are:
    "detector" : the science frame
    "sca" : frame associated with the SCA
    "gwa" " just before the GWA going from detector to sky
    "slit_frame" : frame associated with the virtual slit
    "msa_frame" : at the MSA
    "oteip" : after the FWA
    "v2v3" : at V2V3
    "world" : sky and spectral

    Parameters
    ----------
    input_model : `~jwst.datamodels.JwstDataModel`
        The input data model.
    reference_files : dict
        The reference files used for this mode.
    slit_y_range : list
        The slit dimensions relative to the center of the slit.
    """
    open_slits_id = get_open_slits(input_model, reference_files, slit_y_range)
    if not open_slits_id:
        return None
    n_slits = len(open_slits_id)
    log.info("Computing WCS for {0} open slitlets".format(n_slits))

    msa_pipeline = slitlets_wcs(input_model, reference_files, open_slits_id)

    return msa_pipeline

def slitlets_wcs(input_model, reference_files, open_slits_id):
    """
    Create The WCS pipeline for MOS and Fixed slits for the
    specific opened shutters/slits. ``slit_y_range`` is taken from
    ``slit.ymin`` and ``slit.ymax``.

    Note: This function is also used by the ``msaflagopen`` step.
    """
    # Get the corrected disperser model
    disperser = nirspec.get_disperser(input_model, reference_files['disperser'])

    # Get the default spectral order and wavelength range and record them in the model.
    sporder, wrange = nirspec.get_spectral_order_wrange(input_model, reference_files['wavelengthrange'])
    input_model.meta.wcsinfo.waverange_start = wrange[0]
    input_model.meta.wcsinfo.waverange_end = wrange[1]
    log.info("SPORDER= {0}, wrange={1}".format(sporder, wrange))
    input_model.meta.wcsinfo.spectral_order = sporder

    # DMS to SCA transform
    dms2detector = nirspec.dms_to_sca(input_model)
    dms2detector.name = 'dms2sca'
    # DETECTOR to GWA transform
    det2gwa = Identity(2) & nirspec.detector_to_gwa(reference_files,
                                                    input_model.meta.instrument.detector,
                                                    disperser)
    det2gwa.name = "det2gwa"

    # GWA to SLIT
    gwa2slit = nirspec.gwa_to_slit(open_slits_id, input_model, disperser, reference_files)
    gwa2slit.name = "gwa2slit"

    # SLIT to MSA transform
    slit2msa = nirspec.slit_to_msa(open_slits_id, reference_files['msa'])
    slit2msa.name = "slit2msa"

    # Create coordinate frames in the NIRSPEC WCS pipeline"
    # "detector", "gwa", "slit_frame", "msa_frame", "oteip", "v2v3", "v2v3vacorr", "world"
    det, sca, gwa, slit_frame, msa_frame, oteip, v2v3, v2v3vacorr, world = create_frames()

    exp_type = input_model.meta.exposure.type.upper()

    is_lamp_exposure = exp_type in ['NRS_LAMP', 'NRS_AUTOWAVE', 'NRS_AUTOFLAT']

    if input_model.meta.instrument.filter == 'OPAQUE' or is_lamp_exposure:
        # convert to microns if the pipeline ends earlier
        msa_pipeline = [(det, dms2detector),
                        (sca, det2gwa),
                        (gwa, gwa2slit),
                        (slit_frame, slit2msa),
                        (msa_frame, None)]
    else:
        # MSA to OTEIP transform
        msa2oteip = nirspec.msa_to_oteip(reference_files)
        msa2oteip.name = "msa2oteip"

        # OTEIP to V2,V3 transform
        # This includes a wavelength unit conversion from meters to microns.
        oteip2v23 = nirspec.oteip_to_v23(reference_files, input_model)
        oteip2v23.name = "oteip2v23"

        # Compute differential velocity aberration (DVA) correction:
        va_corr = pointing.dva_corr_model(
            va_scale=input_model.meta.velocity_aberration.scale_factor,
            v2_ref=input_model.meta.wcsinfo.v2_ref,
            v3_ref=input_model.meta.wcsinfo.v3_ref
        ) & Identity(1)

        # V2, V3 to sky
        tel2sky = pointing.v23tosky(input_model) & Identity(1)
        tel2sky.name = "v2v3_to_sky"

        msa_pipeline = [(det, dms2detector),
                        (sca, det2gwa),
                        (gwa, gwa2slit),
                        (slit_frame, slit2msa),
                        (msa_frame, msa2oteip),
                        (oteip, oteip2v23),
                        (v2v3, va_corr),
                        (v2v3vacorr, tel2sky),
                        (world, None)]

    return msa_pipeline



def maskgen(ra,dec):
    """
    Input coordinates and figure out where they will fall on the MSA shutter array
    and where in the respective shutters
    """

    # Need transformation from ra/dec to the shutter plane

    # Need list of broken open and closed shutters

    # assign_wcs
    # need gWCS object
    # assign_wcs.load_wcs()

    instrument = 'nirspec'
    mod = importlib.import_module('.nirspec','jwst.assign_wcs')
    # from jwst.assign_wcs import nirspec
    wcs = mod.slits_wcs(input_model, reference_files, slit_y_range)

    data = datamodels.ImageModel()
    data.meta.instrument.name = 'NIRSPEC'
    data.meta.instrument.grating = 'G140H'
    data.meta.instrument.filter = 'F100LP'
    data.meta.instrument.detector = 'NRS2'
    data.meta.instrument.gwa_pxav = 179.1740788000057
    data.meta.instrument.gwa_pyav = 66.73139400000171
    data.meta.instrument.gwa_tilt = 36.06178279999935
    data.meta.instrument.gwa_xp_v = 179.1855152
    data.meta.instrument.gwa_xtilt = 0.3576895300000106
    data.meta.instrument.gwa_yp_v = 66.731394
    data.meta.instrument.gwa_ytilt = 0.1332439180000042
    data.meta.instrument.msa_state = 'PRIMARYPARK_CONFIG'
    data.meta.instrument.msa_metadata_file = 'jw02609009001_04_msa.fits'
    data.meta.instrument.msa_metadata_id = 313
    data.meta.observation.date = '2024-10-22'
    data.meta.observation.time = '22:31:26.597'
    data.meta.exposure.type = 'NRS_MSASPEC'
    data.meta.dither.position_number = 1

    
    reference_files = []
    slit_y_range = 100
    dum = mod.slits_wcs(data, reference_files, slit_y_range)

    from jwst.assign_wcs.assign_wcs_step import AssignWcsStep
    dum = AssignWcsStep()
    out = dum.get_reference_file(input_file, reference_file_type)

    # get_reference_file() is actually from stpipe.step
    # https://github.com/spacetelescope/stpipe/blob/main/src/stpipe/step.py

    reference_file_types = ['distortion', 'filteroffset', 'specwcs', 'regions',
                            'wavelengthrange', 'camera', 'collimator', 'disperser',
                            'fore', 'fpa', 'msa', 'ote', 'ifupost',
                            'ifufore', 'ifuslicer']

    reference_file_names = {}
    for reftype in reference_file_types:
        reffile = crds_client.get_reference_file(data.get_crds_parameters(), reftype, 'jwst')
        reference_file_names[reftype] = reffile if reffile else ""
    
    from stpipe import crds_client
    reference_name = crds_client.get_reference_file(data.get_crds_parameters(),
                                                    'specwcs','jwst')

    #reference_name = crds_client.get_reference_file(model.get_crds_parameters(),
    #                                                reference_file_type,
    #                                                model.crds_observatory,
    #                                                )

    msa_metadata_file, msa_metadata_id, dither_point = mod.get_msa_metadata(data, reference_file_names)

    msa_config = reference_files['msametafile']
    
    
    dum = slits_wcs(data,reference_file_names,slit_y_range=[-.55, .55])
    
    # Get the corrected disperser model
    disperser = mod.get_disperser(data, reference_file_names['disperser'])
    disperser.groovedensity = disperser.groove_density
    
    # DMS to SCA transform
    dms2detector = mod.dms_to_sca(data)

    # DETECTOR to GWA transform
    det2gwa = mod.detector_to_gwa(reference_files, input_model.meta.instrument.detector, disperser)

    # Get the default spectral order and wavelength range and record them in the model.
    sporder, wrange = get_spectral_order_wrange(input_model, reference_files['wavelengthrange'])
    input_model.meta.wcsinfo.waverange_start = wrange[0]
    input_model.meta.wcsinfo.waverange_end = wrange[1]
    input_model.meta.wcsinfo.spectral_order = sporder


    det2world = slit.meta.wcs.get_transform('detector','world')
    rr2,dd2,wcs_wl2 = det2world(xw,yw)
    det2slit = slit.meta.wcs.get_transform('detector','slit_frame')
    xs,ys,ws = det2slit(xw,yw)
    
    
    # The coordinate frames are:
    # "detector" : the science frame
    # "sca" : frame associated with the SCA
    # "gwa" " just before the GWA (grating wheel assembly) going from detector to sky
    # "slit_frame" : frame associated with the virtual slit
    # "msa_frame" : at the MSA
    # "oteip" : after the FWA
    # "v2v3" : at V2V3
    # "world" : sky and spectral


    # detector    x/y position on the detector
    # sca         sensor chip assembly
    # gwa         grating wheel assembly
    # slit_frame  in the relevant slit, x=0 and y=0 are the center.
    #               x=[-1,1], y=[-1,1] or is it [-0.5,0.5]?
    # msa_frame   micro-shutter array
    # oteip       after the FWA (filter wheel assembly)
    # v2v3        instrument-independent focal plane reference frame
    # world       sky and spectral
    # collimator  ??
    # dms          In "DMS" orientation (same parity as the sky),
    #               DMS coordinates

    # assign_wcs.slits_wcs()
    #    "detector" : the science frame
    #    "sca" : frame associated with the SCA (sensor chip assembly)
    #    "gwa" " just before the GWA (grating wheel assembly) going from detector to sky
    #    "slit_frame" : frame associated with the virtual slit
    #    "msa_frame" : at the MSA
    #    "oteip" : after the FWA (filter wheel assembly)
    #    "v2v3" : at V2V3
    #    "world" : sky and spectral

    # from gwcs import coordinate_frames as cf
    # from gwcs import wcs
    # frame : `~coordinate_frames.CoordinateFrame`

    # https://gwcs.readthedocs.io/en/latest/api/gwcs.coordinate_frames.CoordinateFrame.html

    #stdatamodels.jwst.transforms.models
    #https://stdatamodels.readthedocs.io/en/latest/jwst/transforms/index.html
    from stdatamodels.jwst.transforms import models
    help(models)

    
    
def traces(ra,dec,filt='F100LP',grating='G140H'):
    """
    Input coordinates and figure out where they fall on the detector and their
    full traces.
    """
    
    # Also want to transform to the detector plane

    instrument = 'nirspec'    
    
