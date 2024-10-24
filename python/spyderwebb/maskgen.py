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
from jwst.assign_wcs import pointing,nirspec,load_wcs,assign_wcs
from stpipe import crds_client
from astropy.modeling import models

# Generate JWST NIRSpec MSA masks


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

    input_model = datamodels.ImageModel()
    input_model.meta.instrument.name = 'NIRSPEC'
    input_model.meta.instrument.grating = 'G140H'
    input_model.meta.instrument.filter = 'F100LP'
    input_model.meta.instrument.detector = 'NRS2'
    input_model.meta.instrument.gwa_pxav = 179.1740788000057
    input_model.meta.instrument.gwa_pyav = 66.73139400000171
    input_model.meta.instrument.gwa_tilt = 36.06178279999935
    input_model.meta.instrument.gwa_xp_v = 179.1855152
    input_model.meta.instrument.gwa_xtilt = 0.3576895300000106
    input_model.meta.instrument.gwa_yp_v = 66.731394
    input_model.meta.instrument.gwa_ytilt = 0.1332439180000042
    input_model.meta.instrument.msa_state = 'PRIMARYPARK_CONFIG'
    input_model.meta.instrument.msa_metadata_file = 'jw02609009001_04_msa.fits'
    input_model.meta.instrument.msa_metadata_id = 313
    input_model.meta.observation.date = '2024-10-22'
    input_model.meta.observation.time = '22:31:26.597'
    input_model.meta.exposure.type = 'NRS_MSASPEC'
    input_model.meta.dither.position_number = 1
    input_model.meta.wcsinfo.ra_ref = 11.54924651643522
    input_model.meta.wcsinfo.dec_ref = 42.0959801450496
    input_model.meta.wcsinfo.v2_ref = 378.563202
    input_model.meta.wcsinfo.v3_ref = -428.402832
    input_model.meta.wcsinfo.v3yangle = 138.5745697
    input_model.meta.wcsinfo.roll_ref = 89.30712467134603
    input_model.meta.wcsinfo.velosys = 18316.66
    input_model.meta.wcsinfo.vparity = -1
    

    #from jwst.assign_wcs.assign_wcs_step import AssignWcsStep
    #dum = AssignWcsStep()
    #out = dum.get_reference_file(input_file, reference_file_type)

    # get_reference_file() is actually from stpipe.step
    # https://github.com/spacetelescope/stpipe/blob/main/src/stpipe/step.py

    reference_file_types = ['distortion', 'filteroffset', 'specwcs', 'regions',
                            'wavelengthrange', 'camera', 'collimator', 'disperser',
                            'fore', 'fpa', 'msa', 'ote', 'ifupost',
                            'ifufore', 'ifuslicer']
    reference_file_names = {}
    for reftype in reference_file_types:
        reffile = crds_client.get_reference_file(input_model.get_crds_parameters(), reftype, 'jwst')
        reference_file_names[reftype] = reffile if reffile else ""

    slit_y_range = [-.55, .55]

    msa_pipeline = nirspec.slits_wcs(input_model,reference_file_names,slit_y_range)
    msa_wcs = WCS(msa_pipeline)

    # msa_wcs.available_frames
    # ['detector', 'sca', 'gwa', 'slit_frame', 'msa_frame', 'oteip', 'v2v3', 'v2v3vacorr', 'world']
    
    # load_wcs() creates a gWCS object and stores it in input_model.meta
    result = assign_wcs.load_wcs(input_model, reference_file_names, slit_y_range)
    # in load_wcs()
    # from gwcs.wcs import WCS
    # wcs = WCS(pipeline)
    # output_model.meta.wcs = wcs
    
    imaging_pipeline = nirspec.imaging(input_model, reference_file_names)
    imaging_wcs = WCS(imaging_pipeline)

    # imaging_wcs.available_frames
    # ['detector', 'sca', 'gwa', 'msa', 'oteip', 'v2v3', 'v2v3vacorr', 'world']
    
    world2msa = imaging_wcs.get_transform('world','msa')

    dum = world2msa(11.54924651643522,42.0959801450496)

    dum = world2msa([11.54924651643522,11.548],[42.0959801450496,42.0960])
    # (array([9.90022962e-06, 9.07088640e-04]),
    #  array([ 3.60360417e-05, -8.95968281e-04]))
    # what units are these, how do I convert it to a quadrant and row/column?

    from astropy.io import ascii,fits
    tab = ascii.read('M31-b19-rgb-fields1278.cat')
    x,y = world2msa(tab['ra'],tab['dec'])


    # slits msa wcs pipeline
    #  msa_pipeline = [(det, dms2detector),
    #                  (sca, det2gwa),
    #                  (gwa, gwa2slit),
    #                  (slit_frame, slit2msa),
    #                  (msa_frame, msa2oteip),
    #                  (oteip, oteip2v23),
    #                  (v2v3, va_corr),
    #                  (v2v3vacorr, tel2sky),
    #                  (world, None)]

    # imaging wcs pipeline
    #    imaging_pipeline = [(det, dms2detector),
    #                        (sca, det2gwa),
    #                        (gwa, gwa2msa),
    #                        (msa_frame, msa2oteip),
    #                        (oteip, oteip2v23),
    #                        (v2v3, va_corr),
    #                        (v2v3vacorr, tel2sky),
    #                        (world, None)]

        
    # There is a nirspec.slits_to_msa() function
    # that returns a Slit2Msa() object, but you have to give it
    # a list of open lists

    # slitdata_model = nirspec.get_slit_location_model(slitdata)

    # def get_slit_location_model(slitdata):
    #    slitdata : ndarray
    #    An array of shape (5,) with elements:
    #    slit_id, xcenter, ycenter, xsize, ysize
    #    This is the slit info in the MSa description file.
    #     num, xcenter, ycenter, xsize, ysize = slitdata
    #     model = models.Scale(xsize) & models.Scale(ysize) | \
    #         models.Shift(xcenter) & models.Shift(ycenter)
    #     return model


    # hdu=fits.open('jw02609009001_04_msa.fits')
    # hdu[2].data
    #FITS_rec([(  2,   1, 2,  16,  68, 201177, 'N', 'OPEN', 0.138, 0.32 , 1, 'Y'),
    #          (  3,   1, 1,  17,  50, 119245, 'N', 'OPEN', 0.539, 0.665, 1, 'Y'),
    #          (  4,   1, 1,  18,  97, 112916, 'N', 'OPEN', 0.564, 0.756, 1, 'Y'),
    # dtype=(numpy.record, [('slitlet_id', '>i2'), ('msa_metadata_id', '>i2'), ('shutter_quadrant', '>i2'),
    #    ('shutter_row', '>i2'), ('shutter_column', '>i2'), ('source_id', '>i4'), ('background', 'S1'),
    #    ('shutter_state', 'S6'), ('estimated_source_in_shutter_x', '>f4'),
    #    ('estimated_source_in_shutter_y', '>f4'), ('dither_point_index', '>i2'), ('primary_source', 'S1')]))

    # want this information for all of the shutters


    import asdf
    # reference_file_names['msa']
    msadata = asdf.open('/home/x51j468/crds_cache/references/jwst/nirspec/jwst_nirspec_msa_0011.asdf')
    # msadata['Q1']['data']

    slitdata = msadata['Q1']['data'][0]
    num, xcenter, ycenter, xsize, ysize = slitdata
    model = models.Scale(xsize) & models.Scale(ysize) | \
        models.Shift(xcenter) & models.Shift(ycenter)

    # in nirspec.slit_to_msa()
    msa_quadrant = msadata['Q1']
    msa_model = msa_quadrant.model  # CompoundModel
    slitdata_model = get_slit_location_model(slitdata)
    msa_transform = slitdata_model | msa_model

    xc1 = np.array([o[1] for o in msadata['Q1']['data']])
    yc1 = np.array([o[2] for o in msadata['Q1']['data']])
    # 62415

    xc2 = np.array([o[1] for o in msadata['Q2']['data']])
    yc2 = np.array([o[2] for o in msadata['Q2']['data']])

    xc3 = np.array([o[1] for o in msadata['Q3']['data']])
    yc3 = np.array([o[2] for o in msadata['Q3']['data']])

    xc4 = np.array([o[1] for o in msadata['Q4']['data']])
    yc4 = np.array([o[2] for o in msadata['Q4']['data']])

    # all of the 4 sets of coordinates overlap
    # so they are relative to the "origin" of their respective quadrant
    # values from x=0.0-0.038 to y=0.0-0.0346
    
    # msadata['Q1']['model']
    # <CompoundModel(angle_0=-0.02501392, offset_1=-0.04269253, offset_2=-0.04176972)>

    # msadata['Q2']['model']
    # <CompoundModel(angle_0=-0.01731757, offset_1=-0.04280757, offset_2=0.00725996)>

    # msadata['Q3']['model']
    # <CompoundModel(angle_0=-0.01353264, offset_1=0.00459343, offset_2=-0.04177503)>

    # msadata['Q4']['model']
    # <CompoundModel(angle_0=-0.01337407, offset_1=0.00460352, offset_2=0.00724746)>

    
    plt.scatter(xc1-0.04269253,yc1-0.04176972,s=5)
    plt.scatter(xc2-0.04280757,yc2+0.00725996,s=5)
    plt.scatter(xc3+0.00459343,yc3-0.04177503,s=5)
    plt.scatter(xc4+0.00460352,yc4+0.00724746,s=5)
    # that looks about right

    # now the stars on top, seems about right
    plt.scatter(x,y,s=20,marker='+')

    
    # you can also use nirspec.create_pipeline() to get a wcs pipeline using the EXP_TYPE
    
    # but how do I use this WCS "pipeline"???

    
    #msa_metadata_file, msa_metadata_id, dither_point = mod.get_msa_metadata(input_model, reference_file_names)

    #msa_config = reference_files['msametafile']
    
    

    
    # Get the corrected disperser model
    disperser = nirspec.get_disperser(input_model, reference_file_names['disperser'])
    disperser.groovedensity = disperser.groove_density
    
    # DMS to SCA transform
    dms2detector = nirspec.dms_to_sca(input_model)

    # DETECTOR to GWA transform
    det2gwa = nirspec.detector_to_gwa(reference_files, input_model.meta.instrument.detector, disperser)

    # Get the default spectral order and wavelength range and record them in the model.
    sporder, wrange = nirspec.get_spectral_order_wrange(input_model, reference_files['wavelengthrange'])
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
    #from stdatamodels.jwst.transforms import models
    #help(models)

    # I might want "imaging" transform if I'm just interested in the MSA position
    # intead of "slits_wcs"
    exp_type2transform = {
        'nrs_autoflat':  slits_wcs,
        'nrs_autowave':  nrs_lamp,
        'nrs_brightobj': slits_wcs,
        'nrs_confirm':   imaging,
        'nrs_dark':      not_implemented_mode,
        'nrs_fixedslit': slits_wcs,
        'nrs_focus':     imaging,
        'nrs_ifu':       ifu,
        'nrs_image':     imaging,
        'nrs_lamp':      nrs_lamp,
        'nrs_mimf':      imaging,
        'nrs_msaspec':   slits_wcs,
        'nrs_msata':     imaging,
        'nrs_taconfirm': imaging,
        'nrs_tacq':      imaging,
        'nrs_taslit':    imaging,
        'nrs_verify':    imaging,
        'nrs_wata':      imaging,
    }
    
    
def traces(ra,dec,filt='F100LP',grating='G140H'):
    """
    Input coordinates and figure out where they fall on the detector and their
    full traces.
    """
    
    # Also want to transform to the detector plane

    instrument = 'nirspec'    
    
