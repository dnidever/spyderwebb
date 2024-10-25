import os
import numpy as np
import json
from gwcs.wcs import WCS
import asdf
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
from jwst.assign_wcs import pointing,nirspec,assign_wcs
from stpipe import crds_client
from astropy.modeling import models
from datetime import datetime
from astropy.time import Time
from . import utils

# Generate JWST NIRSpec MSA masks

MSA2SLIT_COEF = np.array([[[ 4.07470547e+02,  9.52495217e+03, -4.15835866e+00],
                           [ 2.06309383e+02,  2.14492760e+00,  4.91307616e+03]],
                          [[ 4.08780083e+02,  9.52539935e+03, -2.87903952e+00],
                           [-3.46119764e+01,  1.48525568e+00,  4.91401850e+03]],
                          [[-4.28486131e+01,  9.52546438e+03, -2.24981142e+00],
                           [ 2.06226763e+02,  1.16034868e+00,  4.91279401e+03]],
                          [[-4.28366954e+01,  9.52592872e+03, -2.22355660e+00],
                           [-3.46210821e+01,  1.14709013e+00,  4.91424362e+03]]])

class NIRSpecTransform(object):

    def __init__(self,grating='G140H',filt='F100LP',ra=11.549246,dec=42.095980,
                 pa_aper=0.0,msafile=None):

        input_model = create_initial_nirspec_model(ra,dec,grating,filt,pa_aper,msafile=msafile)
        self.input_model = input_model
        self._ra = ra
        self._dec = dec
        
        reference_file_types = ['distortion', 'filteroffset', 'specwcs', 'regions',
                                'wavelengthrange', 'camera', 'collimator', 'disperser',
                                'fore', 'fpa', 'msa', 'ote', 'ifupost',
                                'ifufore', 'ifuslicer', 'msaoper']
        reference_file_names = {}
        for reftype in reference_file_types:
            reffile = crds_client.get_reference_file(input_model.get_crds_parameters(), reftype, 'jwst')
            reference_file_names[reftype] = reffile if reffile else ""

        self.reference_file_types = reference_file_types
        self.reference_file_names = reference_file_names
        
        slit_y_range = [-.55, .55]
        self.slit_y_range = slit_y_range
        
        msa_pipeline = nirspec.slits_wcs(input_model,reference_file_names,slit_y_range)
        self.msa_pipeline = msa_pipeline
        msa_wcs = WCS(msa_pipeline)
        self.msa_wcs = msa_wcs
        #self._world2detector = msa_wcs.get_transform('world','detector')
        
        # msa_wcs.available_frames
        # ['detector', 'sca', 'gwa', 'slit_frame', 'msa_frame', 'oteip', 'v2v3', 'v2v3vacorr', 'world']
    
        # load_wcs() creates a gWCS object and stores it in input_model.meta
        result = assign_wcs.load_wcs(input_model, reference_file_names, slit_y_range)
        # in load_wcs()
        # from gwcs.wcs import WCS
        # wcs = WCS(pipeline)
        # output_model.meta.wcs = wcs
    
        imaging_pipeline = nirspec.imaging(input_model, reference_file_names)
        self.imaging_pipeline = imaging_pipeline
        imaging_wcs = WCS(imaging_pipeline)
        self.imaging_wcs = imaging_wcs
        
        # imaging_wcs.available_frames
        # ['detector', 'sca', 'gwa', 'msa', 'oteip', 'v2v3', 'v2v3vacorr', 'world']
    
        world2msa = imaging_wcs.get_transform('world','msa')
        self._world2msa = world2msa

        # Load shutter status information
        with open(reference_file_names['msaoper']) as f:
            shutterinfo = json.load(f)
        dt = [('Q',int),('x',int),('y',int),('state',str,10),('TAstate',str,10),
              ('Internalstate',str,10),('Vignetted',str,10)]
        shutterstate = np.zeros(len(shutterinfo['msaoper']),dtype=np.dtype(dt))
        shutterid = np.zeros(len(shutterinfo['msaoper']),(str,10))
        for i,a in enumerate(shutterinfo['msaoper']):
            shutterstate['Q'][i] = a['Q']
            shutterstate['x'][i] = a['x']
            shutterstate['y'][i] = a['y']
            shutterstate['state'][i] = a['state']                    # "closed" or "open"
            shutterstate['TAstate'][i] = a['TA state']               # "closed" or "open"
            shutterstate['Internalstate'][i] = a['Internal state']   # "closed", "open" or "normal"
            shutterstate['Vignetted'][i] = a['Vignetted']            # "no" or "yes"
            shutterid[i] = str(a['Q'])+'-'+str(a['x'])+'-'+str(a['y'])
        self._shutterstate = shutterstate
        # shutter ID for quick lookup, q-c-r
        self._shutterstate_id = shutterid

        # Get MSA model information
        self.msa = asdf.open(self.reference_file_names['msa'])
        
            
    def world2msa(self,ra,dec):
        """ Convert ra/dec to coordiantes in the MSA frame."""
        return self._world2msa(ra,dec)

    def world2slit(self,ra,dec):
        """ Get slit quadrant/row/column information for in put ra/dec coordinates."""
        # xmsa/ymsa have units of meters in the MSA plane/frame
        xmsa,ymsa = self._world2msa(ra,dec)
        quadrant,column,row,xslit,yslit = msa2slit(xmsa,ymsa)
        return quadrant,column,row,xslit,yslit

    def update_refcoords(self,ra,dec):
        """ Change the reference coordinates."""
        # quickly updates the input_model and all the wcs pipelines and objects
        self._ra = ra
        self.input_model.meta.wcsinfo.ra_ref = ra
        self._dec = dec
        self.input_model.meta.wcsinfo.dec_ref = dec
        # imaging wcs pipeline
        #    imaging_pipeline = [(det, dms2detector),
        #                        (sca, det2gwa),
        #                        (gwa, gwa2msa),
        #                        (msa_frame, msa2oteip),
        #                        (oteip, oteip2v23),
        #                        (v2v3, va_corr),
        #                        (v2v3vacorr, tel2sky),
        #                        (world, None)]
        # We only need to update tel2sky
        tel2sky = pointing.v23tosky(self.input_model)
        self.imaging_pipeline[-2] = (self.imaging_pipeline[-2][0],tel2sky)
        # Remake imaging_wcs
        self.imaging_wcs = WCS(self.imaging_pipeline)
        # Remake world2msa
        self._world2msa = self.imaging_wcs.get_transform('world','msa')
        
        # slits msa wcs pipeline
        # msa_pipeline = [(det, dms2detector),
        #                 (sca, det2gwa),
        #                 (gwa, gwa2slit),
        #                 (slit_frame, slit2msa),
        #                 (msa_frame, msa2oteip),
        #                 (oteip, oteip2v23),
        #                 (v2v3, va_corr),
        #                 (v2v3vacorr, tel2sky),
        #                 (world, None)]
        self.msa_pipeline[-2] = (self.msa_pipeline[-2][0],tel2sky)
        # Remake msa_wcs
        self.msa_wcs = WCS(self.msa_pipeline)
        # Remake world2detector
        self._world2detector = self.msa_wcs.get_transform('world','detector')
        
    @property
    def ra(self):
        return self._ra

    @property
    def dec(self):
        return self._dec

    def shutterstate(self,quadrant,column,row):
        """ Return the state of the requested shutters shutters"""
        squadrant = np.atleast_1d(quadrant).astype(str)
        scolumn = np.atleast_1d(column).astype(str)
        srow = np.atleast_1d(row).astype(str)
        shutterid = np.char.array(squadrant)+'-'+np.char.array(scolumn)+'-'+np.char.array(srow)
        _,ind1,ind2 = np.intersect1d(shutterid,self._shutterstate_id,return_indices=True)
        # Initialize the output tables
        dt = [('Q',int),('x',int),('y',int),('status',str,10),('state',str,10),
              ('TAstate',str,10),('Internalstate',str,10),('Vignetted',str,10)]        
        state = np.zeros(len(squadrant),dtype=np.dtype(dt))
        state['Q'] = quadrant
        state['x'] = column
        state['y'] = row
        state['status'] = 'okay'
        if len(ind1)>0:
            state['status'][ind1] = 'broken'
            for c in self._shutterstate.dtype.names:
                state[c][ind1] = self._shutterstate[c][ind2]
        return state

    def gwa_to_slit(self,quadrant,column,row):
        """ Get gwa2slit transformation for a single slit."""
        agreq = nirspec.angle_from_disperser(self.disperser, self.input_model)
        collimator2gwa = nirspec.collimator_to_gwa(self.reference_file_names, self.disperser)
        lgreq = nirspec.wavelength_from_disperser(self.disperser, self.input_model)
        try:
            velosys = self.input_model.meta.wcsinfo.velosys
        except AttributeError:
            pass
        else:
            if velosys is not None:
                velocity_corr = nirspec.velocity_correction(self.input_model.meta.wcsinfo.velosys)
                lgreq = lgreq | velocity_corr
                log.info("Applied Barycentric velocity correction : {}".format(velocity_corr[1].amplitude.value))

        # The wavelength units up to this point are
        # meters as required by the pipeline but the desired output wavelength units is microns.
        # So we are going to Scale the spectral units by 1e6 (meters -> microns)
        is_lamp_exposure = self.input_model.meta.exposure.type in ['NRS_LAMP', 'NRS_AUTOWAVE', 'NRS_AUTOFLAT']
        if self.input_model.meta.instrument.filter == 'OPAQUE' or is_lamp_exposure:
            lgreq = lgreq | Scale(1e6)

        msa_quadrant = self.msa['Q'+str(quadrant)]
        msa_model = msa_quadrant.model
        msa_data = msa_quadrant.data
        # Slit('S1600A1', 3, 0, 0, 0, slit_y_range[0], slit_y_range[1], 5, 1)
        #slit = models.Slit(slitlet_id, shutter_id, dither_position,
        #                   xcen, ycen, ymin, ymax, quadrant, source_id,
        #                   all_shutters, source_name, source_alias,
        #                   stellarity, source_xpos, source_ypos,
        #                   source_ra, source_dec)
        #slit = Slit('S1600A1', s+1, 0, 0, 0, slit_y_range[0], slit_y_range[1], q, 1)
        slit = Slit('S1600A1', s+1, 0, 0, 0, self.slit_y_range[0], self.slit_y_range[1], quadrant, 1)
        # slit_id is a running number from 1 to 62415
        slit_id = (row-1)*365 + column   # CHECK THIS!!
        slit.shutter_id = slit_id
        
        #mask = mask_slit(slit.ymin, slit.ymax)
        #slit_id = slit.shutter_id
        # Shutter IDs are numbered starting from 1
        # while FS are numbered starting from 0.
        # "Quadrant 5 is for fixed slits.
        #if quadrant != 5:
        #    slit_id -= 1
        slitdata = msa_data[slit_id-1]
        slitdata_model = nirspec.get_slit_location_model(slitdata)
        msa_transform = (slitdata_model | msa_model)
        msa2gwa = (msa_transform | collimator2gwa)
        gwa2msa = nirspec.gwa_to_ymsa(msa2gwa, slit=None, slit_y_range=self.slit_y_range)  # TODO: Use model sets here
        #gwa2msa = nirspec.gwa_to_ymsa(msa2gwa, slit=slit, slit_y_range=(slit.ymin, slit.ymax))  # TODO: Use model sets here
        bgwa2msa = Mapping((0, 1, 0, 1), n_inputs=3) | \
            Const1D(0) * Identity(1) & Const1D(-1) * Identity(1) & Identity(2) | \
            Identity(1) & gwa2msa & Identity(2) | \
            Mapping((0, 1, 0, 1, 2, 3)) | Identity(2) & msa2gwa & Identity(2) | \
            Mapping((0, 1, 2, 3, 5), n_inputs=7) | Identity(2) & lgreq | mask
        #   Mapping((0, 1, 2, 5), n_inputs=7) | Identity(2) & lgreq | mask
        # and modify lgreq to accept alpha_in, beta_in, alpha_out
        # msa to before_gwa
        msa2bgwa = msa2gwa & Identity(1) | Mapping((3, 0, 1, 2)) | agreq
        bgwa2msa.inverse = msa2bgwa
        #slit_models.append(bgwa2msa)
        #slits.append(slit)
        return Gwa2Slit([slit], [bgwa2msa])

    def slit_to_msa(self,quadrant,column,row):
        #def slit_to_msa(open_slits, msafile):
        """
        The transform from ``slit_frame`` to ``msa_frame``.
        
        Parameters
        ----------
        open_slits : list
           A list of slit IDs for all open shutters/slitlets.
        msafile : str
          The name of the msa reference file.

        Returns
        -------
        model : `~stdatamodels.jwst.transforms.Slit2Msa` model.
           Transform from ``slit_frame`` to ``msa_frame``.
        """
        msa_quadrant = self.msa['Q'+str(quadrant)]
        msa_data = msa_quadrant.data
        msa_model = msa_quadrant.model
        slit = Slit('S1600A1', s+1, 0, 0, 0, self.slit_y_range[0], self.slit_y_range[1], quadrant, 1)
        # slit_id is a running number from 1 to 62415
        slit_id = (row-1)*365 + column   # CHECK THIS!!
        slit.shutter_id = slit_id
        # Shutters are numbered starting from 1.
        # Fixed slits (Quadrant 5) are mapped starting from 0.
        #if quadrant != 5:
        #    slit_id = slit_id - 1
        slitdata = msa_data[slit_id-1]
        slitdata_model = nirspec.get_slit_location_model(slitdata)
        msa_transform = slitdata_model | msa_model
        #models.append(msa_transform)
        #slits.append(slit)
        return Slit2Msa([slit], [msa_transform])
    
    def world2detector(self,ra,dec):
        quadrant,column,row,xslit,yslit = self.world2slit(ra,dec)
        # the msa_pipeline/wcs is configured for particular slits
        # need to generate new wcs for each slit
        n = np.atleast_1d(quadrant).size
        for i in range(n):
            msa_pipeline = self.msa_pipeline.copy()
            # msa_pipeline = [(det, dms2detector),
            #                 (sca, det2gwa),
            #                 (gwa, gwa2slit),
            #                 (slit_frame, slit2msa),
            #                 (msa_frame, msa2oteip),
            #                 (oteip, oteip2v23),
            #                 (v2v3, va_corr),
            #                 (v2v3vacorr, tel2sky),
            #                 (world, None)]
            # gwa2slit and slit2msa need to be modified
            gwa2slit = msa_pipeline[2][1]
            slit2msa = msa_pipeline[3][1]            

            # GWA to SLIT
            gwa2slit = self.gwa_to_slit(quadrant,column,row)
            #gwa2slit = nirspec.gwa_to_slit(open_slits_id, self.input_model, disperser, self.reference_files_names)
            gwa2slit.name = "gwa2slit"

            # SLIT to MSA transform
            slit2msa = self.slit_to_msa(quadrant,column,row)
            #slit2msa = nirspec.slit_to_msa(open_slits_id, self.reference_files['msa'])
            slit2msa.name = "slit2msa"
            
            
    def traces(self,ra,dec):
        """ Determine the traces on the detector."""
        return self.msa_wcs(ra,dec)
    
def slitpositions(msa):
    """
    Get MSA positions for all of the slits/shutters
    """

    slit_y_range = [-.55, .55]

    sdata = []
    # Loop over quadrants
    for q in range(1,5):
        msa_quadrant = msa['Q'+str(q)]
        msa_model = msa_quadrant['model']
        msa_data = msa_quadrant['data']
        for s in range(len(msa_data)):
            slitdata = msa_data[s]
            num, xcenter, ycenter, xsize, ysize = slitdata
            # Slit('S1600A1', 3, 0, 0, 0, slit_y_range[0], slit_y_range[1], 5, 1)
            #slit = models.Slit(slitlet_id, shutter_id, dither_position,
            #                   xcen, ycen, ymin, ymax, quadrant, source_id,
            #                   all_shutters, source_name, source_alias,
            #                   stellarity, source_xpos, source_ypos,
            #                   source_ra, source_dec)
            #slit = Slit('S1600A1', s+1, 0, 0, 0, slit_y_range[0], slit_y_range[1], q, 1)
            
            slitdata_model = models.Scale(xsize) & models.Scale(ysize) | \
                models.Shift(xcenter) & models.Shift(ycenter)
            msa_transform = slitdata_model | msa_model
            x,y = msa_transform(0,0)
            #slit2msa = Slit2Msa([slit], [msa_transform])
            ##x,y = slit2msa('S1600A1',0,0)
            #x,y = slit2msa.models[0](0,0)
            sdata.append((num, xcenter, ycenter, xsize, ysize, q, x, y))
            print(sdata[-1])

    dt = [('num',int),('xcenter',float),('ycenter',float),('xsize',float),
          ('ysize',float),('quadrant',int),('xmsa',float),('ymsa',float)]
    data = np.zeros(len(sdata),dtype=np.dtype(dt))
    data[...] = sdata
    
    return sdata

def msa2slit(xmsa,ymsa):
    """
    Convert xmsa/ymsa to slit quadrant/row/column
    """
    # xmsa/ymsa have units of meters in the MSA plane/frame
    # [4,2,3],  [quadrant, row/column, coefficients]
    quadrant = (xmsa > 0)*2 + (ymsa > 0)*1 + 1
    ccoef = MSA2SLIT_COEF[quadrant-1,0,:]
    rcoef = MSA2SLIT_COEF[quadrant-1,1,:]
    cc = ccoef[:,0] + xmsa*ccoef[:,1] + ymsa*ccoef[:,2]
    rr = rcoef[:,0] + xmsa*rcoef[:,1] + ymsa*rcoef[:,2]
    # convert to integer row/column and decimal xslit/yslit
    column = np.round(cc).astype(int)
    row = np.round(rr).astype(int)
    # xslit/yslit values go from 0.0 to 1.0, 0.5 is the center of the shutter/slit
    xslit = cc-column + 0.5
    yslit = rr-row + 0.5
    # Note that the observation msa fits file has row and column reversed
    
    return quadrant,column,row,xslit,yslit

def create_initial_nirspec_model(ra,dec,grating,filt,pa_aper=0.0,msafile=None):
    """
    Initialize a NIRSpec MSA data model
    """
    if grating not in ['G140H','G140M','G235H','G235M','G395H','G395M']:
        raise Exception(grating+' not supported')
    
    instrument = 'nirspec'
    input_model = datamodels.ImageModel()
    input_model.meta.instrument.name = 'NIRSPEC'
    input_model.meta.instrument.grating = grating
    input_model.meta.instrument.filter = filt
    input_model.meta.instrument.detector = 'NRS2'

    # GWA_XTILT:  Grating wheel tilt along instrmnt model Y axis
    #   Grating wheel tilt sensor value GWA_X_TILT_AVGED, corrected for sensor supply voltage
    #   offsets and averaged over multiple samples. In the instrument/wheel-centric reference
    #   frame this is the angle ALONG the X axis, while in the NIRSpec instrument model (used
    #   to define the WCS of the data) this is the angle AROUND the X axis, and hence is
    #   equivalent to the angle along the Y axis.
    # GWA_YTILT:  Grating wheel tilt along instrmnt model X axis
    #   Grating wheel tilt sensor value GWA_Y_TILT_AVGED, corrected
    #   for sensor supply voltage offsets and averaged over multiple
    #   samples. In the instrument/wheel-centric reference frame this
    #   is the angle ALONG the Y axis, while in the NIRSpec instrument
    #   model (used to define the WCS of the data) this is the angle AROUND
    #   the Y axis, and hence is equivalent to the angle along the X axis.
    # GWA_XP_V:  GWA X-position/tilt sensor calibrated
    #   Grating wheel X-position tilt sensor calibrated value.
    # GWA_YP_V:  GWA Y-position/tilt sensor calibrated
    #   Grating wheel Y-position tilt sensor calibrated value.
    # GWA_PXAV:  GWA REC avg X-pos/tilt sensor calibrated
    #   Grating wheel REC-averaged and calibrated X-position tilt sensor value.
    # GWA_PYAV:  GWA REC avg Y-pos/tilt sensor calibrated
    #   Grating wheel REC-averaged and calibrated Y-position tilt sensor value.
    # GWA_TILT:  GWA TILT (avg/calibrated) temperature
    #   Grating wheel temperature sensor calibrated physical value.
    
    # GWA and other parameters depend on the grating
    # There is still 
    if grating=='G140H':
        #input_model.meta.instrument.gwa_pxav = 179.1740788000057
        #input_model.meta.instrument.gwa_pyav = 66.73139400000171
        #input_model.meta.instrument.gwa_tilt = 36.06178279999935
        #input_model.meta.instrument.gwa_xp_v = 179.1855152
        #input_model.meta.instrument.gwa_xtilt = 0.3576895300000106
        #input_model.meta.instrument.gwa_yp_v = 66.731394
        #input_model.meta.instrument.gwa_ytilt = 0.1332439180000042

        input_model.meta.instrument.gwa_pxav = 181.46136
        input_model.meta.instrument.gwa_pyav = 66.97156
        input_model.meta.instrument.gwa_tilt = 36.08218
        input_model.meta.instrument.gwa_xp_v = 181.43849
        input_model.meta.instrument.gwa_xtilt = 0.36216
        input_model.meta.instrument.gwa_yp_v = 66.96012
        input_model.meta.instrument.gwa_ytilt = 0.13367
        coef = np.array([1.00019019, -138.9041342])
        roll_ref = np.polyval(coef,pa_aper)
        input_model.meta.wcsinfo.roll_ref = roll_ref
    elif grating=='G140M':
        input_model.meta.instrument.gwa_pxav = 165.03869
        input_model.meta.instrument.gwa_pyav = 62.43131
        input_model.meta.instrument.gwa_tilt = 36.09452
        input_model.meta.instrument.gwa_xp_v = 165.05012
        input_model.meta.instrument.gwa_xtilt = 0.32940
        input_model.meta.instrument.gwa_yp_v = 62.43131
        input_model.meta.instrument.gwa_ytilt = 0.12462
        coef = np.array([1.00009524, -138.93522129])
        roll_ref = np.polyval(coef,pa_aper)
        input_model.meta.wcsinfo.roll_ref = roll_ref
    elif grating=='G235H':
        input_model.meta.instrument.gwa_pxav = 177.64160
        input_model.meta.instrument.gwa_pyav = 69.05298
        input_model.meta.instrument.gwa_tilt = 36.07978
        input_model.meta.instrument.gwa_xp_v = 177.65304
        input_model.meta.instrument.gwa_xtilt = 0.35456
        input_model.meta.instrument.gwa_yp_v = 69.04155
        input_model.meta.instrument.gwa_ytilt = 0.13785
        coef = np.array([1.00031184, -138.85961624])
        roll_ref = np.polyval(coef,pa_aper)
        input_model.meta.wcsinfo.roll_ref = roll_ref
    elif grating=='G235M':
        input_model.meta.instrument.gwa_pxav = 158.48563
        input_model.meta.instrument.gwa_pyav = 69.32746
        input_model.meta.instrument.gwa_tilt = 36.09692
        input_model.meta.instrument.gwa_xp_v = 158.46276
        input_model.meta.instrument.gwa_xtilt = 0.31635
        input_model.meta.instrument.gwa_yp_v = 69.32746
        input_model.meta.instrument.gwa_ytilt = 0.13840
        coef = np.array([1.00011506, -138.93469585])
        roll_ref = np.polyval(coef,pa_aper)
        input_model.meta.wcsinfo.roll_ref = roll_ref
    elif grating=='G395H':
        input_model.meta.instrument.gwa_pxav = 159.00027
        input_model.meta.instrument.gwa_pyav = 72.49534
        input_model.meta.instrument.gwa_tilt = 36.12278
        input_model.meta.instrument.gwa_xp_v = 159.01171
        input_model.meta.instrument.gwa_xtilt = 0.31735
        input_model.meta.instrument.gwa_yp_v = 72.50678
        input_model.meta.instrument.gwa_ytilt = 0.14471
        coef = np.array([1.00014264, -138.86636817])
        roll_ref = np.polyval(coef,pa_aper)
        input_model.meta.wcsinfo.roll_ref = roll_ref
    elif grating=='G395M':
        input_model.meta.instrument.gwa_pxav = 141.21667
        input_model.meta.instrument.gwa_pyav = 89.02094
        input_model.meta.instrument.gwa_tilt = 36.10675
        input_model.meta.instrument.gwa_xp_v = 141.23382
        input_model.meta.instrument.gwa_xtilt = 0.28186
        input_model.meta.instrument.gwa_yp_v = 89.03237
        input_model.meta.instrument.gwa_ytilt = 0.17767
        coef = np.array([1.00018741, -138.96722442])
        roll_ref = np.polyval(coef,pa_aper)
        input_model.meta.wcsinfo.roll_ref = roll_ref
    else:
        raise Exception(grating+' not supported')
            
    input_model.meta.instrument.msa_state = 'PRIMARYPARK_CONFIG'
    if msafile is None:
        msafile = utils.datadir()+'jw02609009001_04_msa.fits'
    input_model.meta.instrument.msa_metadata_file = msafile
    input_model.meta.instrument.msa_metadata_id = 1
    #msahdu = fits.open(msafile)
    #hdu[2].data['msa_metadata_id'][0]
    # Use today's date
    obsdate = Time.now().isot
    input_model.meta.observation.date = obsdate.split('T')[0]
    input_model.meta.observation.time = obsdate.split('T')[1]
    input_model.meta.exposure.type = 'NRS_MSASPEC'
    input_model.meta.dither.position_number = 1
    input_model.meta.wcsinfo.ra_ref = ra
    input_model.meta.wcsinfo.dec_ref = dec

    # velocity_corr = utils.velocity_correction(input_model.meta.wcsinfo.velosys)

    # Science Instrument Aperture Files (SIAF)
    
    # V2_REF:  [arcsec] Telescope V2 coord of reference point
    #   The telescope V2 coordinate, in units of arc seconds, at the aperture reference
    #   point. Taken from the SIAF 'V2Ref' entry
    # V3_REF:  [arcsec] Telescope V3 coord of reference point
    #   The telescope V3 coordinate, in units of arc seconds, at the aperture reference
    #   point. Taken from the SIAF 'V3Ref' entry
    # V3I_YANGLE (V3YANGLE):  Direction angle in V3 (Y)
    #   Angle from V3 axis to Ideal y axis (deg)
    #input_model.meta.wcsinfo.v2_ref = 378.563202
    #input_model.meta.wcsinfo.v3_ref = -428.402832
    #input_model.meta.wcsinfo.v3yangle = 138.5745697

    input_model.meta.wcsinfo.v2_ref = 299.26538
    input_model.meta.wcsinfo.v3_ref = -456.72629
    input_model.meta.wcsinfo.v3yangle = 138.57578
    
    # velosys [m/s] Barycentric correction to radial velocity
    #input_model.meta.wcsinfo.velosys = 18316.66
    input_model.meta.wcsinfo.vparity = -1

    #input_model.meta.instrument.msa_metadata_id = 313
    #input_model.meta.observation.date = '2024-10-22'
    #input_model.meta.observation.time = '22:31:26.597'

    
    return input_model

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

    #from astropy.io import ascii,fits
    #tab = ascii.read('M31-b19-rgb-fields1278.cat')
    #x,y = world2msa(tab['ra'],tab['dec'])


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
    
