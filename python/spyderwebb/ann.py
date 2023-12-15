import os
import copy
import time
import numpy as np
import traceback
from astropy.table import Table
from scipy.optimize import curve_fit
from theborg.emulator import Emulator
from doppler.spec1d import Spec1D
from . import utils

cspeed = 2.99792458e5  # speed of light in km/s

class JWSTANNModel():

    # Model JWST nirspec spectra using ANN model
    
    def __init__(self,spobs=None,loggrelation=False,verbose=False):
        # Load the ANN models
        em1 = Emulator.read(utils.datadir()+'ann_29pars_3500-4200.pkl')
        em2 = Emulator.read(utils.datadir()+'ann_29pars_4000-5000.pkl')
        em3 = Emulator.read(utils.datadir()+'ann_29pars_4900-6000.pkl')
        self._models = [em1,em2,em3]
        self.nmodels = len(self._models)
        self.labels = self._models[0].label_names
        self.nlabels = len(self.labels)
        self._ranges = np.zeros((self.nmodels,self.nlabels,2),float)
        for i in range(self.nmodels):
            for j in range(self.nlabels):
                self._ranges[i,j,:] = [np.min(self._models[i].training_labels[:,j]),
                                       np.max(self._models[i].training_labels[:,j])]
        self._ranges[0,0,1] = 4100.0  # use 3500-4200 model up to 4100
        self._ranges[1,0,0] = 4100.0  # use 4000-5000 model from 4100 to 4950
        self._ranges[1,0,1] = 4950.0        
        self._ranges[2,0,0] = 4950.0  # use 4900-6000 model from 4950
        self.ranges = np.zeros((self.nlabels,2),float)
        for i in range(self.nlabels):
            self.ranges[i,:] = [np.max(self._ranges[:,i,0]),np.min(self._ranges[:,i,1])]
        
        # Alpha element index
        alphaindex = []
        for i,l in enumerate(self.labels):
            if l in ['om','cam','mgm','tim','sm','sim']:
                alphaindex.append(i)
        self._alphaindex = np.array(alphaindex)
        
        # Input observed spectrum information
        self._spobs = spobs
            
        # ANN model wavelengths
        npix_model = 22001
        self._dispersion = np.arange(npix_model)*0.5+9000

        # Get logg label
        loggind, = np.where(np.char.array(self.labels).lower()=='logg')
        if len(loggind)==0:                
            raise ValueError('No logg label')
        self.loggind = loggind[0]
        
        # Load the ANN model for logg-relationship
        logg_model = Emulator.load(utils.datadir()+'apogeedr17_rgb_logg_ann.npz')
        self.logg_model = logg_model
        
        self.loggrelation = loggrelation
        self.verbose = verbose
        self.fitparams = None
        self.njac = 0
        
    def mklabels(self,pars):
        """ Make the labels array from a dictionary."""
        # Dictionary input
        if type(pars) is dict:
            labels = np.zeros(28)
            # Must at least have Teff and logg
            for k in pars.keys():
                if k=='alpham':   # mean alpha
                    ind = self._alphaindex.copy()
                else:
                    ind, = np.where(np.array(self.labels)==k.lower())
                if len(ind)==0:
                    raise ValueError(k+' not found in labels: '+','.join(self.labels))
                labels[ind] = pars[k]
            if labels[0]<=0 or labels[1]<=0:
                raise ValueError('pars must at least have teff and logg')
        # List or array input
        else:
            if self.fitparams is not None and len(pars) != len(self.labels):
                if len(pars) != len(self.fitparams):
                    raise ValueError('pars size not consistent with fitparams')
                labels = np.zeros(self.nlabels)
                for i in range(len(pars)):
                    ind, = np.where(np.array(self.labels)==self.fitparams[i])
                    labels[ind] = pars[i]
            else:
                labels = pars
            #if len(labels)<len(self.labels):
            #    raise ValueError('pars must have '+str(len(self.labels))+' elements')

        return labels

    def meanlabels(self):
        """ Return mean labels."""
        return np.mean(self.ranges,axis=1)

    def get_best_model(self,labels):
        """ This returns the first ANN model that has the right range."""
        for m in range(self.nmodels):
            ranges = self._ranges[m,:,:]
            inside = True
            for i in range(3):
                inside &= (labels[i]>=ranges[i,0]) & (labels[i]<=ranges[i,1])
            if inside:
                return m
        return None
    
    def mkbounds(self,params):
        """ Make bounds for input parameter names."""
        bounds = [np.zeros(len(params)),np.zeros(len(params))]
        for i in range(len(params)):
            if params[i].lower()=='alpham':   # mean alpha
                ind = self._alphaindex.copy()
                bounds[0][i] = np.max(self.ranges[ind,0])
                bounds[1][i] = np.min(self.ranges[ind,1])
            else:
                ind, = np.where(np.array(self.labels)==params[i].lower())
                bounds[0][i] = self.ranges[ind,0]
                bounds[1][i] = self.ranges[ind,1]
    
    def inrange(self,pars):
        """ Check that the parameters are in range."""
        labels = self.mklabels(pars)
        # Get the right model to use based on input Teff/logg/feh
        modelindex = self.get_best_model(labels)
        if modelindex is None:
            return False,0,[self.ranges[0,0],self.ranges[0,1]]
        # Check other ranges
        for i in np.arange(1,self.nlabels):
            rr = [self._ranges[modelindex,i,0],self._ranges[modelindex,i,1]]
            if labels[i]<rr[0] or labels[i]>rr[1]:
                return False,i,rr
        return True,None,None

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

    def printpars(self,pars,perror):
        """ Print out parameters and errors."""
        
        for i in range(len(pars)):
            if self.fitparams is not None and len(pars) != self.nlabels:
                name = self.fitparams[i]
            else:
                name = self.labels[i]
            if i==0:
                print('{:6s}: {:10.1f} +/- {:5.2g}'.format(name,pars[i],perror[i]))
            else:
                print('{:6s}: {:10.4f} +/- {:5.3g}'.format(name,pars[i],perror[i]))

    def randompars(self,labels=None,n=100):
        """ Create random parameters for initial guesses."""
        if labels is None:
            labels = self.labels
        nlabels = len(labels)
        pars = np.zeros((n,nlabels),float)
        for i in range(nlabels):
            ind, = np.where(np.array(self.labels)==labels[i])
            vmin = self.ranges[ind,0]
            vmax = self.ranges[ind,1]
            vrange = vmax-vmin
            # make a small buffer
            vmin += vrange*0.01
            vrange *= 0.98
            pars[:,i] = np.random.rand(n)*vrange+vmin

        return pars

    def fiducialspec(self):
        """ Default JWST resolution and wavelength spectrum."""
        # Default observed spectrum            
        wobs_coef = np.array([-1.51930967e-09, -5.46761333e-06,  2.39684716e+00,  8.99994494e+03])            
        # 3847 observed pixels
        npix_obs = 3847
        wobs = np.polyval(wobs_coef,np.arange(npix_obs))
        spobs = Spec1D(np.zeros(npix_obs),wave=wobs,err=np.ones(npix_obs),
                       lsfpars=np.array([ 1.05094118e+00, -3.37514635e-06]),
                       lsftype='Gaussian',lsfxtype='wave')        
        return spobs
    
    def __call__(self,pars=None,spobs=None,snr=None,vrel=None,normalize=False,
                 fiducial=False):
        """
        Returns JWST model spectrum.

        Parameters
        ----------
        pars : list or array
           Parameters for the labels.
        spobs : Spec1D, optional
           Observed spectrum to model, using the resolution and wavelength.
        snr : float, optional
           Add random noise to the spectrum at the "snr" level.
        vrel : float, optional
           Doppler shift the spectrum by vrel (km/s).
        normalize : bool, optional
           Perform continuum normalization on the spectrum.
        fiducial : bool, optional
           Use fiducial JWST resolution and wavelength values.

        Returns
        -------
        spec : Spec1D
           JWST model spectrum.

        Example
        -------

        spec = jw(pars)

        """
        # If no pars input, use mean values
        if pars is None:
            pars = self.meanlabels()
        # Get label array
        labels = self.mklabels(pars)

        # Check that the labels are in range
        flag,badindex,rr = self.inrange(labels)
        if flag==False:
            srr = '[{:.4f},{:.3f}]'.format(*rr)
            error = 'parameters out of range: '
            error += '{:s}={:.4f}'.format(self.labels[badindex],labels[badindex])
            error += ' '+srr
            if spobs is None:
                return np.zeros(self._spobs.size)+1e30
            else:
                return np.zeros(spobs.size)+1e30
            #raise ValueError(error)

        # Get the right model to use based on input Teff/logg/feh
        modelindex = self.get_best_model(labels)
            
        # Get the ANN model spectrum
        flux = self._models[modelindex](labels)
        wave = self._dispersion
        
        # Doppler shift
        if vrel is not None and vrel != 0.0:
            redwave = wave*(1+vrel/cspeed)  # redshift the wavelengths
            orig_flux = flux.copy()
            flux = dln.interp(redwave,flux,wave,extrapolate=False,fill_value=np.nan)
            flux[~np.isfinite(flux)] = 1.0
                
        # Make the model Spec1D object
        spsyn = Spec1D(flux,wave=wave)
        # Say it is normalized
        spsyn.normalized = True
        spsyn._cont = np.ones(spsyn.flux.shape)        
        # Convolve to observed resolution and wavelength
        if fiducial and spobs is None:   # use fiducial values
            spobs = self.fiducialspec()
        if spobs is not None or self._spobs is not None:
            if spobs is None:
                spmonte = spsyn.prepare(self._spobs)
                spmonte.continuum_func = self._spobs.continuum_func
            else:
                spmonte = spsyn.prepare(spobs)
                spmonte.continuum_func = spobs.continuum_func
        # No observed spectrum input
        else:
            spmonte = spsyn
        # Now normalize if requested
        if normalize:
            spmonte.normalize()
        # Add labels to spectrum object
        spmonte.labels = labels
        # Add noise
        if snr is not None:
            spmonte.flux += np.random.randn(*spmonte.err.shape)*1/snr
            spmonte.err += 1/snr
        # Deal with any NaNs
        bd, = np.where(~np.isfinite(spmonte.flux))
        if len(bd)>0:
            spmonte.flux[bd] = 1.0
            spmonte.err[bd] = 1e30
            spmonte.mask[bd] = True
            
        return spmonte
    
    def model(self,wave,*pars,**kwargs):
        """ Model function for curve_fit."""
        if self.verbose:
            print('model: ',pars)
        out = self(pars,**kwargs)
        # Only return the flux
        if isinstance(out,Spec1D):
            return out.flux
        else:
            return out
        
    def jac(self,wave,*args,retmodel=False,**kwargs):
        """
        Method to return Jacobian matrix.
        This includes the contribution of the lookup table.

        Parameters
        ----------
        args : float
            Model parameter values as separate positional input parameters.
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

        jac = JWSTSyn.jac(wave,*pars)

        """

        fullargs = self.mklabels(args)
        
        # logg relation
        #  add a dummy logg value in
        #if self.loggrelation:
        #    fullargs = np.insert(args,self.loggind,0.0)
        #else:
        #    fullargs = args

        if self.verbose:
            print('jac: ',args)

        # Initialize jacobian matrix
        npix = len(wave)
        fjac = np.zeros((npix,len(self.fitparams)),np.float64)
        
        # Loop over parameters
        pars = np.array(copy.deepcopy(args))
        f0 = self.model(wave,*pars,**kwargs)        
        steps = np.zeros(len(self.fitparams))
        for i in range(len(self.fitparams)):
            ind, = np.where(np.array(self.labels)==self.fitparams[i])
            if self.loggrelation and i==self.loggind:
                continue
            targs = np.array(copy.deepcopy(fullargs))
            if ind==0:
                step = 10.0                
            else:
                step = 0.01
            steps[i] = step
            # Check boundaries, if above upper boundary
            #   go the opposite way
            if targs[ind]>self.ranges[ind,1]:
                step *= -1
            targs[ind] += step
            # Remove dummy logg if using logg relation
            if self.loggrelation:
                targs = np.delete(targs,self.loggind)
            #print(i,step,targs)
            f1 = self.model(wave,*targs,**kwargs)
            fjac[:,i] = (f1-f0)/steps[i]
            
        self.njac += 1
            
        return fjac
        
    
    def fit(self,spec,fitparams=None,loggrelation=False,normalize=False,
            initgrid=True,outlier=False,verbose=False):
        """
        Fit an observed spectrum with the ANN models and curve_fit.

        Parameters
        ----------
        spec : Spec1D 
           Spectrum to fit.
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

        tab,info = fit(spec)

        """

        vrel = 0.0
        spec.vrel = vrel
        if verbose:
            print('Vrel: {:.2f} km/s'.format(vrel))
            print('S/N: {:.2f}'.format(spec.snr))
        # Now normalize
        if normalize:
            if spec.normalized==False:
                spec.normalize()

        if fitparams is None:
            fitparams = self.labels
        self.fitparams = np.array(fitparams)
        nfitparams = len(fitparams)

        # Make bounds
        bounds = self.mkbounds(fitparams)
        
        # Run set of ~100 points to get first estimate
        ngrid = 100
        if initgrid:
            
            #if loggrelation:
            #    nsample = 5
            #    tstep = np.ptp(self._ranges[:,0,:])/nsample
            #    tgrid = np.arange(nsample)*tstep+self._ranges[0,0,0]+tstep*0.5
            #    mstep = np.ptp(self._ranges[:,2,:])/nsample
            #    mgrid = np.arange(nsample)*mstep+self._ranges[0,2,0]+mstep*0.5
            #    astep = np.ptp(self._ranges[:,3,:])/nsample
            #    agrid = np.arange(nsample)*astep+self._ranges[0,3,0]+astep*0.5
            #    tgrid2d,mgrid2d,agrid2d = np.meshgrid(tgrid,mgrid,agrid)
            #    gridpars = np.vstack((tgrid2d.flatten(),mgrid2d.flatten(),agrid2d.flatten())).T
            #else:
            #    nsample = 4
            #    tstep = np.ptp(self._ranges[:,0,:])/nsample/1.1
            #    tgrid = np.arange(nsample)*tstep+self._ranges[0,0,0]+tstep*0.5
            #    gstep = np.ptp(self._ranges[:,1,:])/nsample/1.1
            #    ggrid = np.arange(nsample)*gstep+self._ranges[0,1,0]+gstep*0.5            
            #    mstep = np.ptp(self._ranges[:,2,:])/nsample/1.1
            #    mgrid = np.arange(nsample)*mstep+self._ranges[0,2,0]+mstep*0.5
            #    astep = np.ptp(self._ranges[:,3,:])/nsample/1.1
            #    agrid = np.arange(nsample)*astep+self._ranges[0,3,0]+astep*0.5
            #    tgrid2d,ggrid2d,mgrid2d,agrid2d = np.meshgrid(tgrid,ggrid,mgrid,agrid)
            #    gridpars = np.vstack((tgrid2d.flatten(),ggrid2d.flatten(),mgrid2d.flatten(),agrid2d.flatten())).T
            gridpars = self.randompars(self.fitparams,ngrid)
            if verbose:
               print('Testing an initial set of '+str(gridpars.shape[0])+' random parameters')
            
            # Make the models
            for i in range(gridpars.shape[0]):
                tpars1 = {}
                for j in range(len(self.fitparams)):
                    tpars1[self.fitparams[j]] = gridpars[i,j]
                sp1 = self(tpars1)
                if i==0:
                    synflux = np.zeros((gridpars.shape[0],sp1.size),float)
                synflux[i,:] = sp1.flux
            chisqarr = np.sum((synflux-spec.flux)**2/spec.err**2,axis=1)/spec.size
            bestind = np.argmin(chisqarr)
            estimates = gridpars[bestind,:]
        else:
            estimates = np.zeros(len(self.fitparams))
            ind, = np.where(np.array(self.fitparams)=='teff')
            if len(ind)>0:
                estimates[ind] = 4200.0
            ind, = np.where(np.array(self.fitparams)=='logg')
            if len(ind)>0:
                estimates[ind] = 1.5               
            
        if verbose:
            print('Initial estimates: ',estimates)
            
        try:
            pars,pcov = curve_fit(self.model,spec.wave,spec.flux,p0=estimates,
                                  sigma=spec.err,bounds=bounds,jac=self.jac)
            perror = np.sqrt(np.diag(pcov))
            bestmodel = self.model(spec.wave,*pars)
            chisq = np.sum((spec.flux-bestmodel)**2/spec.err**2)/spec.size
            
            # Get full parameters
            if loggrelation:
                fullpars = self.getlogg(pars)
                fullperror = np.insert(perror,self.loggind,0.0)
            else:
                fullpars = pars
                fullperror = perror

            if verbose:
                print('Best parameters:')
                self.printpars(fullpars,fullperror)
                print('Chisq: {:.3f}'.format(chisq))

            # Construct the output dictionary
            out = {'vrel':vrel,'snr':spec.snr,'pars':fullpars,'perror':fullperror,'wave':spec.wave,
                   'flux':spec.flux,'err':spec.err,'model':bestmodel,'chisq':chisq,
                   'loggrelation':loggrelation,'success':True}
            success = True
        except:
            traceback.print_exc()
            success = False
            out = {'success':False}
                
        # Remove outliers and refit
        if success and outlier:
            diff = spec.flux-bestmodel
            med = np.median(diff)
            sig = dln.mad(diff)
            bd, = np.where(np.abs(diff) > 3*sig)
            nbd = len(bd)
            if nbd>0:
                if verbose:
                    print('Removing '+str(nbd)+' outliers and refitting')
                err = spec.err.copy()
                flux = spec.flux.copy()
                err[bd] = 1e30
                flux[bd] = bestmodel[bd]
                # Save original values
                pars0 = pars
                estimates = pars0
                # Run curve_fit
                try:
                    pars,pcov = curve_fit(self.model,spec.wave,flux,p0=estimates,
                                          sigma=err,bounds=bounds,jac=self.jac)
                    perror = np.sqrt(np.diag(pcov))
                    bestmodel = self.model(spec.wave,*pars)
                    chisq = np.sum((flux-bestmodel)**2/err**2)/len(flux)

                    # Get full parameters
                    if loggrelation:
                        fullpars = self.getlogg(pars)
                        fullperror = np.insert(perror,self.loggind,0.0)
                    else:
                        fullpars = pars
                        fullperror = perror

                    if verbose:
                        print('Best parameters:')
                        self.printpars(fullpars,fullperror)
                        print('Chisq: {:.3f}'.format(chisq))                        
                        
                    # Construct the output dictionary
                    out = {'vrel':vrel,'snr':spec.snr,'pars':fullpars,'perror':fullperror,'wave':spec.wave,
                           'flux':spec.flux,'err':spec.err,'mflux':flux,'merr':err,'noutlier':nbd,'model':bestmodel,
                           'chisq':chisq,'loggrelation':loggrelation,'success':True}
                    success = True
                except:
                    traceback.print_exc()
                    success = False
                    out = {'success':False}

        return out
