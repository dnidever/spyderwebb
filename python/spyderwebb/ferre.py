import os
import copy
import numpy as np
import subprocess
import tempfile
import traceback
import shutil
from dlnpyutils import utils as dln
from scipy.optimize import curve_fit
from theborg import emulator
from astropy.table import Table
from doppler.spec1d import Spec1D,continuum
from . import utils

cspeed = 2.99792458e5  # speed of light in km/s

def gridinfo(filename):
    """ Get information about the FERRE grid."""

    if os.path.exists(filename)==False:
        raise FileNotFoundError(filename)
    
    # Read the header information
    f = open(filename,'r')
    header = []
    line = f.readline().strip()
    while line != '/':
        header.append(line)
        line = f.readline().strip()
    f.close()

    out = {'header':header}

    header = np.char.array(header)
    
    # Parse the information
    for h in header[1:]:
        dum = h.split()
        k = dum[0]
        val = dum[2:]
        if k.startswith('COMMENT'):
            val = ' '.join(val)
        else:
            orig = val.copy()
            val = []
            for v in orig:
                v = v.replace("'","")
                if dln.isnumber(v):
                    v = float(v)
                elif v.isnumeric():
                    v = int(v)
                val.append(v)    
        if len(val)==1:
            val = val[0]
        out[k] = val

    if 'N_OF_DIM' in out.keys():
        out['NDIM'] = int(out['N_OF_DIM'])
    if 'NPIX' in out.keys():
        out['NPIX'] = int(out['NPIX'])
    if 'LOGW' in out.keys():
        out['LOGW'] = int(out['LOGW'])
    if 'VACUUM' in out.keys():
        out['VACUUM'] = int(out['VACUUM'])
    if 'N_P' in out.keys():
        n_p = [int(n) for n in out['N_P']]
        out['N_P'] = n_p
    # Get wave
    npix = out['NPIX']
    w0 = out['WAVE'][0]
    dw = out['WAVE'][1]
    wave = np.arange(npix)*dw+w0
    if out['LOGW']==1:
        wave = 10**wave
    out['WAVELENGTH'] = wave
    # Labels
    labels = []
    for i in range(out['NDIM']):
        label1 = out['LABEL('+str(i+1)+')']
        labels.append(label1)
    out['LABELS'] = labels
    # Array for each label
    for i in range(out['NDIM']):
        name = labels[i]
        llim = out['LLIMITS'][i]
        step = out['STEPS'][i]        
        n_p = out['N_P'][i]
        vals = np.arange(n_p)*step+llim
        out[name] = vals
        
    return out


def interp(pars,wave=None,cont=None,ncont=None,grid='jwstgiant4.dat',
           griddir=None,ferresrc=None):
    """
    Interpolate in the FERRE grid.

    Parameters
    ----------
    pars : numpy array
       Array of parameters/labels.  This can be a 2D array [Nstars,Nlabels].
    wave : numpy array, optional
       Output wavelengths.  By default, all wavelengths of the grid are used.
    cont : int, optional
       FERRE "cont" continuum parameter.
    ncont : int, optional
       FERRE "ncont" continuum parameter.
    grid : str, optional
       The FERRE grid name.
    griddir : str, optional
       Directory of the FERRE grids.
    ferresrc : str, optional
       Filename of FERRE executable.

    Returns
    -------
    out : dict
       Output parameters.  Dictionary with 'wave' and 'flux'.  If a 2D parameter
         is input, then 'flux' will be 2D but 'wave' will still be 1D.

    Example
    -------

    out = interp([5000.0, 2.5, -1.2, 0.2])

    """

    if griddir is None:
        griddir = '/Users/nidever/synspec/winter2017/jwst/'
    gridfile = griddir+grid
    if ferresrc is None:
        ferresrc = '/Users/nidever/projects/ferre/bin/ferre.x'
    info = gridinfo(gridfile)

    # Number of objects
    if np.array(pars).ndim==2:
        nobj = pars.shape[0]
    else:
        nobj = 1
    
    # Read the header information
    f = open(gridfile,'r')
    header = []
    line = f.readline().strip()
    while line != '/':
        header.append(line)
        line = f.readline().strip()
    f.close()
    
    # Set up temporary directory
    tmpdir = tempfile.mkdtemp(prefix='ferre')
    curdir = os.path.abspath(os.curdir)
    os.chdir(tmpdir)
    
    # Create fitting input file
    gridbase = os.path.basename(gridfile)    
    os.symlink(gridfile,tmpdir+'/'+gridbase)
    os.symlink(gridfile.replace('.dat','.unf'),tmpdir+'/'+gridbase.replace('.dat','.unf'))
    os.symlink(gridfile.replace('.dat','.hdr'),tmpdir+'/'+gridbase.replace('.dat','.hdr'))
    lines = []
    lines += ["&LISTA"]
    lines += ["NDIM = 4"]
    lines += ["NOV = 0"]
    lines += ["INDV = 1 2 3 4"]
    lines += ["SYNTHFILE(1) = '"+gridbase+"'"]
    lines += ["F_FORMAT = 1"]
    lines += ["PFILE = 'ferre.ipf'"]
    lines += ["FFILE = 'ferre.frd'"]
    lines += ["OFFILE = 'ferre.mdl'"]    # output best-fit models
    if wave is not None:
        lines += ["WFILE = 'ferre.wav'"]
        lines += ["WINTER = 2"]    # wavelength interpolate the model fluxes
    lines += ["NOBJ = "+str(nobj)]
    if cont is not None:
        lines += ["CONT = "+str(cont)]      # Running mean normalization
        lines += ["NCONT = "+str(ncont)]     # Npixel for running mean
    lines += ["ERRBAR = 1"]
    lines += ["/"]
    dln.writelines('input.nml',lines)

    # Write the .wav file
    if wave is not None:
        wlines = ''.join(['{:14.5E}'.format(w) for w in wave]) 
        dln.writelines('ferre.wav',wlines)
        
    # Write the IPF file
    if np.array(pars).ndim==2:
        flines = []
        for i in range(nobj):
            flines += ['test{:d} {:.3f} {:.3f} {:.3f} {:.3f}'.format(i+1,*pars[i,:])]
    else:
        nstars = 1
        flines = 'test1 {:.3f} {:.3f} {:.3f} {:.3f}'.format(*pars)
    dln.writelines('ferre.ipf',flines)
    
    # Run FERRE
    if os.path.exists('ferre.mdl'): os.remove('ferre.mdl')
    try:
        result = subprocess.check_output([ferresrc],stderr=subprocess.STDOUT)
    except:
        #traceback.print_exc()
        pass
    
    # Read the output
    mlines = dln.readlines('ferre.mdl')
    if nobj==1:
        mlines = mlines[0]
        mflux = np.array(mlines.split()).astype(float)
    else:
        npix = len(np.array(mlines[0].split()).astype(float))
        mflux = np.zeros((nobj,npix),float)
        for i in range(nobj):
            mflux[i,:] = np.array(mlines[i].split()).astype(float)
        
    # Load the wavelengths
    if wave is None:
        wave = info['WAVELENGTH'].copy()

    out = {'wave':wave,'flux':mflux}
        
    # Delete temporary files and directory
    os.chdir(curdir)
    shutil.rmtree(tmpdir)

    return out
    

class FERRE(object):

    def __init__(self,grid='jwstgiant4.dat',outwave=None,cnorder=6,cperclevel=90.0,cbinsize=0.1,
                 loggrelation=False,verbose=False):
        """
        outwave: output wavelength array.  By default, the full grid wavelength array will be used.
        cnorder: continuum normalization polynomial order  Default is 6.
        cperclevel: continuum normalization percentile level for bins.  Default is 90.
        cbinsize: continuum normalization fractional wavelength range (from -1 to 1) to bin.  Default is 0.1.
        """
        gridfile = '/Users/nidever/synspec/winter2017/jwst/'+grid
        ferre = '/Users/nidever/projects/ferre/bin/ferre.x'
        info = gridinfo(gridfile)
        self.grid = grid
        self.gridfile = gridfile
        self.ferre = ferre
        self.info = info
        self.labels = info['LABELS']
        self.nlabels = len(self.labels)
        self.npix = self.info['NPIX']
        self.ranges = np.zeros((self.nlabels,2),float)
        for i in range(self.nlabels):
            self.ranges[i,0] = np.min(self.info[self.labels[i]])
            self.ranges[i,1] = np.max(self.info[self.labels[i]])
        self.steps = self.info['STEPS']
        self.npoints = self.info['N_P']
        self.nspectra = np.sum(self.npoints)
        self.vacuum = self.info['VACUUM']
        self.resolution = self.info['RESOLUTION']
        if outwave is not None:
            self.outwave = outwave
        else:
            self.outwave = None
        self.cnorder = cnorder
        self.cperclevel = cperclevel
        self.cbinsize = cbinsize
        self.loggrelation = loggrelation
        self.verbose = verbose
        self.ncall = 0
        self.njac = 0
        
        # Use logg relation
        if loggrelation:
            # Get logg label
            loggind, = np.where(np.char.array(self.labels).lower()=='logg')
            if len(loggind)==0:                
                raise ValueError('No logg label')
            self.loggind = loggind[0]
            # Get temperature label
            teffind, = np.where(np.char.array(self.labels).lower()=='teff')
            if len(teffind)==0:
                teffind, = np.where(np.char.array(self.labels).lower().find('temp')>-1)
            self.teffind = teffind[0]
            # Get metallicity label
            fehind, = np.where(np.char.array(self.labels).lower()=='feh')
            if len(fehind)==0:
                fehind, = np.where(np.char.array(self.labels).lower()=='metal')
            if len(fehind)==0:
                fehind, = np.where(np.char.array(self.labels).lower()=='mh')                
            if len(fehind)==0:
                fehind, = np.where(np.char.array(self.labels).lower()=='[fe/h]')
            if len(fehind)==0:
                fehind, = np.where(np.char.array(self.labels).lower()=='[m/h]')                
            if len(fehind)==0:
                raise ValueError('No metallicity label')
            self.fehind = fehind[0]
            # Get alpha abundance label
            alphaind, = np.where(np.char.array(self.labels).lower()=='[alpha/fe]')
            if len(alphaind)==0:
                alphaind, = np.where(np.char.array(self.labels).lower()=='alphafe')
            if len(alphaind)==0:
                alphaind, = np.where(np.char.array(self.labels).lower()=='alpha_fe')
            if len(alphaind)==0:
                alphaind, = np.where(np.char.array(self.labels).lower().find('alpha')>-1)
            if len(alphaind)==0:
                self.alphaind = None
            else:
                self.alphaind = alphaind[0]
            # Load the ANN model
            logg_model = emulator.Emulator.load(utils.datadir()+'apogeedr17_rgb_logg_ann.npz')
            self.logg_model = logg_model
            
            
    def __repr__(self):
        """ String representation."""
        out = self.__class__.__name__+'({:s}, {:d} labels, {:d} spectra, {:d} pixels)\n'.format(self.grid,self.nlabels,
                                                                                               self.nspectra,self.npix)
        for i in range(self.nlabels):
            out += '{:8.2f} <= {:^12s} <= {:8.2f}, step {:8.2f} [{:d}]\n'.format(self.ranges[i,0],self.labels[i],
                                                                                 self.ranges[i,1],self.steps[i],self.npoints[i])
        out += '{:8.2f} <= {:^12s} <= {:8.2f}, step {:8.2f} [{:d}]\n'.format(np.min(self.info['WAVELENGTH']),
                                                                             'WAVE',np.max(self.info['WAVELENGTH']),
                                                                             self.info['WAVE'][1],self.npix)
        return out 
        
    def __call__(self,pars,wave=None,cnorder=None,cperclevel=None,cbinsize=None,norm=True):
        """
        pars: array of parameters.  Can be 2D [Nstars,Nlabels].
        """
        if wave is None:
            wave = self.outwave
        if cnorder is None:
            cnorder = self.cnorder
        if cperclevel is None:
            cperclevel = self.cperclevel
        if cbinsize is None:
            cbinsize = self.cbinsize
            
        pars = np.array(pars)
        if pars.ndim==2:
            ninputlabels = pars.shape[1]
        else:
            ninputlabels = len(pars)
        if (ninputlabels != self.nlabels and self.loggrelation==False) or \
           (ninputlabels != (self.nlabels-1) and self.loggrelation==True):
            raise ValueError('Number of labels not correct.')
        
        # Use the logg relation (as a function of Teff/feh/alpha)
        if self.loggrelation:
            # Multiple parameters
            if pars.ndim==2:
                nobj = np.array(pars).shape[0]
                newpars = np.zeros((nobj,self.nlabels),float)
                for i in range(nobj):
                    newpars[i,:] = self.getlogg(pars[i,:])
            # Single set of labels
            else:
                nobj = 1
                newpars = self.getlogg(pars)
        else:
            # Multiple parameters
            if pars.ndim==2:
                nobj = np.array(pars).shape[0]
            else:
                nobj = 1
            newpars = pars

        self.ncall += 1
        out = interp(newpars,wave=wave)

        # Now do the continuum normalization
        if cnorder is not None and norm==True:
            wave = out['wave']
            flux = out['flux'].copy()
            if nobj==1:
                out['flux'] = self.normalize(wave,flux)
            else:
                for i in range(nobj):
                    out['flux'][i,:] = self.normalize(wave,flux[i,:])

        return out
        
    def getlogg(self,pars):
        """ Get logg from the logg relation and fill it into the parameter array where it belongs."""
        # The only parameters should be input, with the logg one missing/excluded
        # Insert a dummy value for logg
        newpars = np.insert(pars,self.loggind,0.0)
        teff = newpars[self.teffind]
        feh = newpars[self.fehind]
        if self.alphaind is not None:
            alpha = newpars[self.alphaind]
        else:
            alpha = 0.0
        logg = self.logg_model([teff,feh,alpha],border='extrapolate')
        newpars[self.loggind] = logg
        return newpars

    def normalize(self,wave,flux,norder=None,perclevel=None,binsize=None):
        """ Normalize a spectrum."""
        if norder is None:
            norder = self.cnorder
        if perclevel is None:
            perclevel = self.cperclevel
        if binsize is None:
            binsize = self.cbinsize
        sp = Spec1D(flux,wave=wave,err=flux*0)
        cont = continuum(sp,norder=norder,perclevel=perclevel,binsize=binsize)
        newflux = flux/cont
        return newflux
    
    def model(self,wave,*pars,**kwargs):
        """ Model function for curve_fit."""
        if self.verbose:
            print('model: ',pars)
        out = self(pars,wave=wave,**kwargs)
        # Only return the flux
        return out['flux']

    def jac(self,wave,*args,retmodel=False,allpars=False,**kwargs):
        """
        Method to return Jacobian matrix.
        This includes the contribution of the lookup table.

        Parameters
        ----------
        wave : numpy array
            Wavelength array.
        args : float
            Model parameter values as separate positional input parameters,
            [amp, xcen, ycen, sky]. If allpars=True, then the model
            parameters are added at the end, i.e. 
            [amp, xcen, ycen, sky, model parameters].
        retmodel : boolean, optional
            Return the model as well.  Default is retmodel=False.

        Returns
        -------
        if retmodel==False
        jac : numpy array
          Jacobian matrix of partial derivatives [N,Npars].
        model : numpy array
          Array of (1-D) model values for the input xdata and parameters.
          If retmodel==True, then (model,jac) are returned.

        Example
        -------

        jac = ferre.jac(xdata,*pars)
        """

        # logg relation
        #  add a dummy logg value in
        if self.loggrelation:
            fullargs = np.insert(args,self.loggind,0.0)
        else:
            fullargs = args

        if self.verbose:
            print('jac: ',args)
            
        # Loop over parameters
        pars = [np.array(copy.deepcopy(args))]
        steps = []
        for i in range(self.nlabels):
            if self.loggrelation and i==self.loggind:
                continue
            targs = np.array(copy.deepcopy(fullargs))
            step = 0.1*self.steps[i]
            # Check boundaries, if above upper boundary
            #   go the opposite way
            if targs[i]>self.ranges[i,1]:
                step *= -1
            targs[i] += step
            # Remove dummy logg if using logg relation
            if self.loggrelation:
                targs = np.delete(targs,self.loggind)
            pars.append(targs)
            steps.append(step)

        # Reformat pars into arrays
        pars = np.array(pars)
        steps = np.array(steps)
            
        # Run FERRE
        out = self(pars,wave=wave)
        flux = out['flux']
        npix = flux.shape[1]
        
        # Initialize jacobian matrix
        fjac = np.zeros((npix,len(steps)),np.float64)
        f0 = flux[0,:]
        for i in range(len(steps)):
            f1 = flux[i+1,:]
            fjac[:,i] = (f1-f0)/steps[i]
            
        self.njac += 1
            
        return fjac

    
def fit(slist,inter=3,algor=1,init=1,indini=None,nruns=1,
        cont=1,ncont=0,errbar=1,grid='jwstgiant4.dat',save=False):
    """ 
    Run FERRE on list of spectra

    Parameters
    ----------
    slist : list
       A list of dictionaries with wave, flux, err and vrel.
    vrel : list
       Input list of doppler shifts for the spectra.
    inter : int, optional
       Interpolation algorithm.
         0   nearest neighbor
         1   linear
         2   quadratic Bezier
         3   cubic Bezier
         4   cubic spline
       Default is 3.
    algor : int, optional
       Search algorithm?
        -1   weighted average over the grid  likely
         0   pixel with lowest value         minlocus
         1   Nelder-mead method (Miller's)   minim
         2   BTR method (Csendes/Miller)     global
         3   uobyqa method (Powell/Miller)    uobyqa
         4   truncated Newton                lmqnbc
         5   MCMC                             mcmcde
       Default is 1.
    init : int, optional
       Starting point for search.
         0   use the values in pfile
         1   follow the rules set by keyword indini
       Default is 1.
    indini : list, optional
       Individual control of starting points for each parameter.
    nruns : int, optional
       Number of searches to be done.  Default is 1.
    cont : int, optional
       Continuum parameter:
         1   polynomial fitting  (ncont order, 0-constant, 1-linear)
         2   segmented normaleization  (ncont segments)
         3   running mean  (ncont pixels)
    ncont : int, optional
       Number for continuum normalization.  If cont=1, then ncont
         gives the polynomial order.  If cont=2, then ncont is the
         number of segements.  If cont=3, then ncont is the number
         of pixels for the running mean.
    errbar : int, optional
       Choice of algorithm to compute error bars.
         0   adopt the distance from solution at which chisq=min(chisq)+1
         1   invert the curvature matrix
       Default is 1.
    save : bool, optional
       Do not delete the temporary directory.  Default is save=False.

    Returns
    -------
    tab : table
       Output table of values.
    info : list
       List of dictionaries, one per spectrum, that includes spectra and arrays.

    Example
    -------

    tab,info = fit(files,vrel,algor=1,nruns=5)

    """
        
    gridfile = '/Users/nidever/synspec/winter2017/jwst/'+grid
    ferre = '/Users/nidever/projects/ferre/bin/ferre.x'

    # Set up temporary directory
    tmpdir = tempfile.mkdtemp(prefix='ferre')
    curdir = os.path.abspath(os.curdir)
    os.chdir(tmpdir)
    print('Running FERRE in temporary directory '+tmpdir)

    # Parameters
    print('INTER = ',inter)
    print('ALGOR = ',algor)
    print('INIT = ',init)
    print('INDINI = ',indini)
    print('NRUNS = ',nruns)
    print('CONT = ',cont)
    print('NCONT = ',ncont)
    print('ERRBAR = ',errbar)
    
    # Create fitting input file
    gridbase = os.path.basename(gridfile)    
    os.symlink(gridfile,tmpdir+'/'+gridbase)
    os.symlink(gridfile.replace('.dat','.unf'),tmpdir+'/'+gridbase.replace('.dat','.unf'))
    os.symlink(gridfile.replace('.dat','.hdr'),tmpdir+'/'+gridbase.replace('.dat','.hdr'))
    lines = []
    lines += ["&LISTA"]
    lines += ["NDIM = 4"]
    lines += ["NOV = 4"]
    lines += ["INDV = 1 2 3 4"]
    lines += ["SYNTHFILE(1) = '"+gridbase+"'"]
    lines += ["F_FORMAT = 1"]
    lines += ["INTER = "+str(inter)]    # cubic Bezier interpolation
    lines += ["ALGOR = "+str(algor)]    # 1-Nelder-Mead,5-MCMC
    lines += ["INIT = "+str(init)]
    if indini is not None:
        lines += ['INDINI = '+np.array2string(np.array(indini)).strip('[]')]        
        nruns = 1
        for term in indini:
            nruns = nruns * term
    #    lines += ["INDINI = "+str(indini)]
    if nruns is not None:
        lines += ["NRUNS = "+str(nruns)]
    lines += ["PFILE = 'ferre.ipf'"]
    lines += ["FFILE = 'ferre.frd'"]
    lines += ["ERFILE = 'ferre.err'"]
    lines += ["WFILE = 'ferre.wav'"]    
    lines += ["OPFILE = 'ferre.opf'"]    # output best-fit parameters and uncertainties
    lines += ["OFFILE = 'ferre.mdl'"]    # output best-fit models
    lines += ["SFFILE = 'ferre.nrd'"]    # normalized data
    lines += ["NOBJ = "+str(len(slist))]
    lines += ["CONT = "+str(cont)]      # Running mean normalization
    lines += ["NCONT = "+str(ncont)]     # Npixel for running mean
    lines += ["WINTER = 2"]    # wavelength interpolate the model fluxes
    lines += ["ERRBAR = "+str(errbar)]
    lines += ["/"]
    dln.writelines('input.nml',lines)
        
    # Put star with maximum npix first
    #  FERRE uses this to set the maximum pixels for ALL the spectra
    for i in range(len(slist)):
        if 'npix' not in slist[i].keys():
            slist[i]['npix'] = len(slist[i]['flux'])
        if 'id' not in slist[i].keys():
            slist[i]['id'] = i+1
    npixall = [s['npix'] for s in slist]
    maxind = np.argmax(npixall)
    npix = np.max(npixall)
    if maxind != 0:
        slist1 = slist.pop(maxind)
        slist = [slist1] + slist

    # Make the list of lines to write out for FERRE
    flines,elines,wlines,plines = [],[],[],[]
    for i in range(len(slist)):
        # Need to buffer the arrays to have consistent number of pixels
        flux = slist[i]['flux']
        err = slist[i]['err']
        wave = slist[i]['wave']        
        if len(flux)<npix:
            nmissing = npix-len(flux)
            flux = np.hstack((flux,np.zeros(nmissing)))
            err = np.hstack((err,np.zeros(nmissing)+1e10))
            wave = np.hstack((wave,np.zeros(nmissing))) 
            
        # Convert to ASCII for FERRE
        obs = ''.join(['{:14.5E}'.format(f) for f in flux])
        obserr = ''.join(['{:14.5E}'.format(e) for e in err])
        obswave = ''.join(['{:14.5E}'.format(w) for w in wave])        
        slist[i]['ferre_fline'] = obs
        slist[i]['ferre_eline'] = obserr
        slist[i]['ferre_wline'] = obswave     
        slist[i]['ferre_pline'] = 'spec'+str(i+1)+' 4000.0  2.5  -0.5  0.1'
        slist[i]['ferre_id'] = 'spec'+str(i+1)
        
        flines.append(slist[i]['ferre_fline'])
        elines.append(slist[i]['ferre_eline'])
        wlines.append(slist[i]['ferre_wline'])
        plines.append(slist[i]['ferre_pline'])
        
    dln.writelines('ferre.frd',flines)
    dln.writelines('ferre.err',elines)
    dln.writelines('ferre.wav',wlines)    
    dln.writelines('ferre.ipf',plines)
    
    # Run FERRE
    if os.path.exists('ferre.mdl'): os.remove('ferre.mdl')
    if os.path.exists('ferre.opf'): os.remove('ferre.opf') 
    print('Running FERRE on '+str(len(slist))+' spectra')
    try:
        fout = open('ferre.log','w')
        out = subprocess.call([ferre],stdout=fout,stderr=subprocess.STDOUT)
        fout.close()
        loglines = dln.readlines('ferre.log')
        #out = subprocess.check_output([ferre])
    except:
        fout.close()        
        #traceback.print_exc()
        pass

    
    # Read the output
    olines = dln.readlines('ferre.opf')
    ids = np.char.array([o.split()[0] for o in olines])
    mlines = dln.readlines('ferre.mdl')
    slines = dln.readlines('ferre.nrd')
    nolines = len(olines)
    print(str(nolines)+' lines in output files')

    # Loop over slist
    for i in range(len(slist)):
        slist1 = slist[i]
        filename = slist1['filename']
        ferre_id = slist1['ferre_id']
        
        # Find the right output line for this star
        ind, = np.where(ids==ferre_id)
        if len(ind)==0:
            print(ferre_id+' not found in output file')
            slist1['success'] = False            
            slist1['pars'] = None
            slist1['parerr'] = None
            slist1['snr2'] = None
            slist1['rchisq'] = None
            slist1['model'] = None
            slist1['smflux'] = None
            continue
        ind = ind[0]
        olines1 = olines[ind]
        mlines1 = mlines[ind]
        slines1 = slines[ind]

        # opfile has name, pars, parerr, fraction of phot data points, log(S/N)^2, log(reduced chisq)
        arr = olines1.split()
        pars = np.array(arr[1:5]).astype(float)
        parerr = np.array(arr[5:9]).astype(float)
        logsnr2 = float(arr[10])
        snr2 = 10**logsnr2
        try:
            logrchisq = float(arr[11])
            rchisq = 10**logrchisq
        except:
            logrchisq = np.nan
            rchisq = np.nan
        slist1['success'] = True            
        slist1['pars'] = pars
        slist1['parerr'] = parerr   
        slist1['snr2'] = snr2
        slist1['rchisq'] = rchisq

        mflux = np.array(mlines1.split()).astype(float)
        smflux = np.array(slines1.split()).astype(float)
        # If this star has less pixels than the first star, then we need to trim some pixels
        #  off the end
        if len(mflux) != slist1['npix']:
            mflux = mflux[0:slist1['npix']]
            smflux = smflux[0:slist1['npix']]
        slist1['model'] = mflux
        slist1['smflux'] = smflux

    # Delete temporary files and directory
    if save==False:    
        os.chdir(curdir)
        shutil.rmtree(tmpdir)

    return slist

def specprep(spec,vrel=None):
    """ Prepare the spectra for input to FERRE """
    # Synspec spectra are in AIR, JWST spectra are in VACUUM, convert
    spec.wavevac = False
    flux = np.array([])
    err = np.array([])
    wave = np.array([])
    mask = np.array([])
    for sp in spec:
        # trim edge pixels
        logood = np.where(sp.mask==False)[0][0]
        higood = np.where(sp.mask==False)[0][-1]
        slc = slice(logood+30,higood-30)
        flux = np.concatenate((flux,sp.flux[slc]))
        err = np.concatenate((err,sp.err[slc]))
        wave = np.concatenate((wave,sp.wave[slc]))
        mask = np.concatenate((mask,sp.mask[slc]))
    # Remove Doppler shift
    if vrel is None:
        if hasattr(spec,'vrel')==False:
            raise ValueError('No vrel to use')
        vrel = spec.vrel
    wave /= (1+vrel/cspeed)
    # Trim wavelengths that are too large
    gd, = np.where(wave < 17990)
    flux = flux[gd]
    err = err[gd]
    wave = wave[gd]
    mask = mask[gd]
    # Grow bad mask by ~20 pixels
    oldmask = mask.copy()
    mask = np.convolve(mask,np.ones(61),mode='same').astype(bool)
    # Interpolate over masked pixels
    xpix = np.arange(len(flux))
    bad, = np.where((mask==True) | (flux<=0))
    good, = np.where((mask==False) & (flux>0))
    if len(bad)>0:
        smfilt = utils.nanmedfilt(flux[good],201)
        newpix = dln.interp(xpix[good],smfilt,xpix[bad],kind='linear')
        flux[bad] = newpix
        #flux[bad] = 0.0
        err[bad] = 1e10

    # Put all the information into slist
    out = {}
    out['npix'] = len(flux)
    out['flux'] = flux
    out['err'] = err
    out['wave'] = wave
    return out


def cfit(slist,vrel,cont=1,ncont=0,loggrelation=False,grid='jwstgiant4.dat',
         initgrid=True,outlier=True,verbose=False):
    """ 
    Fit spectrum with curve_fit running FERRE to get the models
    for each set of parameters.

    Parameters
    ----------
    slist : list
       A list of dictionaries with wave, flux, err and vrel.
    vrel : list
       Input list of doppler shifts for the spectra.
    cont : int, optional
       Continuum parameter:
         1   polynomial fitting  (ncont order, 0-constant, 1-linear)
         2   segmented normaleization  (ncont segments)
         3   running mean  (ncont pixels)
    ncont : int, optional
       Number for continuum normalization.  If cont=1, then ncont
         gives the polynomial order.  If cont=2, then ncont is the
         number of segements.  If cont=3, then ncont is the number
         of pixels for the running mean.
    loggrelation : bool, optional
       Use the logg-relation as a function of Teff/feh/alpha.

    Returns
    -------
    tab : table
       Output table of values.
    info : list
       List of dictionaries, one per spectrum, that includes spectra and arrays.

    Example
    -------

    tab,info = cfit(files,vrel)

    """


    ## Loop over the files
    #slist = []
    #for i in range(nfiles):
    #    filename = files[i]
    #    vrel1 = vrel[i]
    #    if filename is None:
    #        continue
    #    if os.path.exists(filename)==False:
    #        print(filename+' not found')
    #        continue
    #    if os.path.getsize(filename)==0:
    #        print(filename+' is empty')
    #        continue
    #    spec = doppler.read(filename)
    #    spec.vrel = vrel1
    #    #spec.normalize()
    #    slist1 = {'filename':filename,'vrel':vrel1,'snr':spec.snr,'spec':spec}
    #    slist.append(slist1)
    #
    #if len(slist)==0:
    #    print('No spectra to fit')
    #    return None,None

    if type(slist) is not list:
        slist = [slist]
    nspectra = len(slist)
    
    # Loop over spectra
    out = []
    for i in range(nspectra):
        if verbose:
            if i>0: print(' ')
            print('Star '+str(i+1))
            print('-------')
        slist1 = slist[i]
        vrel1 = vrel[i]
        spec = slist1['spec']
        spec.vrel = vrel1
        if verbose:
            print('Vrel: {:.2f} km/s'.format(vrel1))
            print('S/N: {:.2f}'.format(spec.snr))
        # Initialize the FERRE object
        fr = FERRE(loggrelation=loggrelation,verbose=(verbose>1))                
        # Prepare spectrum for FERRE
        pspec = specprep(spec)
        # Now normalize
        newflux = fr.normalize(pspec['wave'],pspec['flux'])
        cont = pspec['flux']/newflux
        pspec['oflux'] = pspec['flux']
        pspec['oerr'] = pspec['err']        
        pspec['flux'] = newflux
        pspec['err'] = pspec['oerr']/cont      
        # Normalize
        fr.outwave = pspec['wave']
        if loggrelation:
            estimates = [4500.0,-1.0,0.0]
            bounds = [np.delete(fr.ranges[:,0],1),np.delete(fr.ranges[:,1],1)]            
        else:
            estimates = [4500.0,2.5,-1.0,0.0]
            bounds = [fr.ranges[:,0],fr.ranges[:,1]]


        # Run grid of ~100 points to get first estimate
        if initgrid:
            if loggrelation:
                nsample = 5
                tstep = np.ptp(fr.ranges[0,:])/nsample
                tgrid = np.arange(nsample)*tstep+fr.ranges[0,0]+tstep*0.5
                mstep = np.ptp(fr.ranges[2,:])/nsample
                mgrid = np.arange(nsample)*mstep+fr.ranges[2,0]+mstep*0.5
                astep = np.ptp(fr.ranges[3,:])/nsample
                agrid = np.arange(nsample)*astep+fr.ranges[3,0]+astep*0.5
                tgrid2d,mgrid2d,agrid2d = np.meshgrid(tgrid,mgrid,agrid)
                gridpars = np.vstack((tgrid2d.flatten(),mgrid2d.flatten(),agrid2d.flatten())).T
            else:
                nsample = 4
                tstep = np.ptp(fr.ranges[0,:])/nsample
                tgrid = np.arange(nsample)*tstep+fr.ranges[0,0]+tstep*0.5
                gstep = np.ptp(fr.ranges[1,:])/nsample
                ggrid = np.arange(nsample)*gstep+fr.ranges[1,0]+gstep*0.5            
                mstep = np.ptp(fr.ranges[2,:])/nsample
                mgrid = np.arange(nsample)*mstep+fr.ranges[2,0]+mstep*0.5
                astep = np.ptp(fr.ranges[3,:])/nsample
                agrid = np.arange(nsample)*astep+fr.ranges[3,0]+astep*0.5  
                tgrid2d,ggrid2d,mgrid2d,agrid2d = np.meshgrid(tgrid,ggrid,mgrid,agrid)
                gridpars = np.vstack((tgrid2d.flatten(),ggrid2d.flatten(),mgrid2d.flatten(),agrid2d.flatten())).T
            if verbose:
                print('Testing an initial grid of '+str(gridpars.shape[0])+' spectra')
            # Run FERRE
            specgrid = fr(gridpars,wave=pspec['wave'])
            chisqarr = np.sum((specgrid['flux']-pspec['flux'])**2/pspec['err']**2,axis=1)/len(pspec['flux'])
            bestind = np.argmin(chisqarr)
            estimates = gridpars[bestind,:]
        if verbose:
            print('Initial estimates: ',estimates)
        
        # Run curve_fit
        try:
            pars,pcov = curve_fit(fr.model,pspec['wave'],pspec['flux'],p0=estimates,
                                  sigma=pspec['err'],bounds=bounds,jac=fr.jac)
            perror = np.sqrt(np.diag(pcov))
            bestmodel = fr.model(pspec['wave'],*pars)
            chisq = np.sum((pspec['flux']-bestmodel)**2/pspec['err']**2)/len(pspec['flux'])

            # Get full parameters
            if loggrelation:
                fullpars = fr.getlogg(pars)
                fullperror = np.insert(perror,fr.loggind,0.0)
            else:
                fullpars = pars
                fullperror = perror

            if verbose:
                printvals = [fullpars[0],fullperror[0],fullpars[1],fullperror[1],fullpars[2],fullperror[2],
                             fullpars[3],fullperror[3]]
                print('Best parameters: {:f}+/-{:.3g}, {:.3f}+/-{:.3g}, {:.3f}+/-{:.3g}, {:.3f}+/-{:.3g}'.format(*printvals))
                print('Chisq: ',chisq)

            # Construct the output dictionary
            out1 = {'index':i,'vrel':vrel1,'snr':spec.snr,'pars':fullpars,'perror':fullperror,'wave':pspec['wave'],
                    'flux':pspec['flux'],'err':pspec['err'],'model':bestmodel,'chisq':chisq,
                    'loggrelation':loggrelation,'success':True}
            success = True
        except:
            traceback.print_exc()
            success = False
            out1 = {'success':False}
                
        # Remove outliers and refit
        if success and outlier:
            diff = pspec['flux']-bestmodel
            med = np.median(diff)
            sig = dln.mad(diff)
            bd, = np.where(np.abs(diff) > 3*sig)
            nbd = len(bd)
            if nbd>0:
                if verbose:
                    print('Removing '+str(nbd)+' outliers and refitting')
                err = pspec['err'].copy()
                flux = pspec['flux'].copy()
                err[bd] = 1e30
                flux[bd] = bestmodel[bd]
                # Save original values
                pars0 = pars
                estimates = pars0
                # Run curve_fit
                try:
                    pars,pcov = curve_fit(fr.model,pspec['wave'],flux,p0=estimates,
                                          sigma=err,bounds=bounds,jac=fr.jac)
                    perror = np.sqrt(np.diag(pcov))
                    bestmodel = fr.model(pspec['wave'],*pars)
                    chisq = np.sum((flux-bestmodel)**2/err**2)/len(flux)

                    # Get full parameters
                    if loggrelation:
                        fullpars = fr.getlogg(pars)
                        fullperror = np.insert(perror,fr.loggind,0.0)
                    else:
                        fullpars = pars
                        fullperror = perror

                    if verbose:
                        printvals = [fullpars[0],fullperror[0],fullpars[1],fullperror[1],fullpars[2],fullperror[2],
                                     fullpars[3],fullperror[3]]
                        print('Best parameters: {:f}+/-{:f}, {:.3f}+/-{:.3f}, {:.3f}+/-{:.3f}, {:.3f}+/-{:.3f}'.format(*printvals))
                        print('Chisq: ',chisq)
                        
                    # Construct the output dictionary
                    out1 = {'index':i,'vrel':vrel1,'snr':spec.snr,'pars':fullpars,'perror':fullperror,'wave':pspec['wave'],
                            'flux':pspec['flux'],'err':pspec['err'],'mflux':flux,'merr':err,'noutlier':nbd,'model':bestmodel,
                            'chisq':chisq,'loggrelation':loggrelation,'success':True}
                    success = True
                except:
                    traceback.print_exc()
                    success = False
                    out1 = {'success':False}
                
        #if verbose and success:
        #    print('Best parameters: ',out1['pars'])
        #    print('Chisq: ',out1['chisq'])
            
        out.append(out1)

    # Create output table
    dt = [('index',int),('vrel',float),('snr',float),('pars',float,4),('perror',float,4),('chisq',float),('success',bool)]
    tab = np.zeros(len(out),dtype=np.dtype(dt))
    for i,o in enumerate(out):
        if o['success']:
            tab['index'][i] = o['index']
            tab['vrel'][i] = o['vrel']
            tab['snr'][i] = o['snr']                        
            tab['pars'][i] = o['pars']	 
            tab['perror'][i] = o['perror']
            tab['chisq'][i] = o['chisq']
            tab['success'][i] = o['success']
    tab = Table(tab)

        
    return out,tab
