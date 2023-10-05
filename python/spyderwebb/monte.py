import os
import copy
import time
import numpy as np
import traceback
from scipy.optimize import curve_fit
from theborg.emulator import Emulator
from doppler.spec1d import Spec1D
from . import utils

class JWSTSyn():

    # model JWST nirspec spectra
    
    def __init__(self,spobs=None,loggrelation=False,verbose=False):
        # Load the ANN models
        datadir = '/Users/nidever/synspec/nodegrid/'
        emwarm = Emulator.read(datadir+'grid7/grid7_annmodel_300neurons_0.0001rate_20000steps.pkl')
        emcool = Emulator.read(datadir+'grid8/grid8_annmodel_300neurons_0.0001rate_20000steps.pkl')        
        self._models = [emcool,emwarm]
        self.nmodels = len(self._models)
        self.labels = self._models[0].label_names
        self.nlabels = len(self.labels)
        self._ranges = np.zeros((self.nmodels,self.nlabels,2),float)
        for i in range(self.nmodels):
            for j in range(self.nlabels):
                self._ranges[i,j,:] = [np.min(self._models[i].training_labels[:,j]),
                                       np.max(self._models[i].training_labels[:,j])]
        self._ranges[0,0,1] = 4100.0
        self._ranges[1,0,0] = 4100.0        
        self.ranges = np.zeros((self.nlabels,2),float)
        self.ranges[0,:] = [self._ranges[0,0,0],self._ranges[1,0,1]]
        for i in np.arange(1,self.nlabels):
            self.ranges[i,:] = [np.max(self._ranges[:,i,0]),np.min(self._ranges[:,i,1])]
        
        # alpha element indexes
        alphaindex = []
        for i,l in enumerate(self.labels):
            if l in ['om','cam','mgm','tim','sm','sim']:
                alphaindex.append(i)
        self._alphaindex = np.array(alphaindex)
        
        # Input observed spectrum information
        if spobs is not None:
            self._spobs = spobs
        # Default observed spectrum            
        else:
            wobs_coef = np.array([-1.51930967e-09, -5.46761333e-06,  2.39684716e+00,  8.99994494e+03])            
            # 3847 observed pixels
            npix_obs = 3847
            wobs = np.polyval(wobs_coef,np.arange(npix_obs))
            spobs = Spec1D(np.zeros(npix_obs),wave=wobs,err=np.ones(npix_obs),
                           lsfpars=np.array([ 1.05094118e+00, -3.37514635e-06]),
                           lsftype='Gaussian',lsfxtype='wave')        
            self._spobs = spobs

        # Synthetic wavelengths
        npix_syn = 22001
        self._wsyn = np.arange(npix_syn)*0.5+9000

        # Load the ANN model
        logg_model = Emulator.load(utils.datadir()+'apogeedr17_rgb_logg_ann.npz')
        self.logg_model = logg_model
        
        self.loggrelation = loggrelation
        self.verbose = verbose
        self.njac = 0
        
    def mklabels(self,pars):
        """ Make the labels array from a dictionary."""
        # Dictionary input
        if type(pars) is dict:
            labels = np.zeros(28)
            # allow alpha to be input
            # Must at least have Teff and logg
            for k in pars.keys():
                if k=='alpham':
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
            labels = pars
            if len(labels)<len(self.labels):
                raise ValueError('pars must have '+str(len(self.labels))+' elements')

        return labels
        
    def inrange(self,pars):
        """ Check that the parameters are in range."""
        if type(pars) is dict:
            labels = self.mklabels(pars)
        else:
            labels = pars
        # Check temperature
        rr = [self._ranges[0,0,0],self._ranges[1,0,1]]
        if labels[0]<rr[0] or labels[0]>rr[1]:
            return False,0,rr
        # Get the right model to use based on input Teff
        if labels[0] < self._ranges[0,0,1]:
            modelindex = 0
        else:
            modelindex = 1
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
    
    def __call__(self,pars,snr=None,spobs=None):
        # Get label array
        if type(pars) is dict:
            labels = self.mklabels(pars)
        else:
            labels = pars

        # Check that the labels are in range
        flag,badindex,rr = self.inrange(labels)
        if flag==False:
            srr = '[{:.4f},{:.3f}]'.format(*rr)
            error = 'parameters out of range: '
            error += '{:s}={:.4f}'.format(self.labels[badindex],labels[badindex])
            error += ' '+srr
            raise ValueError(error)

        # Get the right model to use based on input Teff
        if labels[0] < self._ranges[0,0,1]:
            modelindex = 0
        else:
            modelindex = 1
            
        # Get the synthetic spectrum
        flux = self._models[modelindex](labels)

        # Make the synthetic Spec1D object
        spsyn = Spec1D(flux,wave=self._wsyn)
        # Say it is normalized
        spsyn.normalized = True
        spsyn._cont = np.ones(spsyn.flux.shape)        
        # Convolve to JWST resolution and wavelength
        if spobs is None:
            spmonte = spsyn.prepare(self._spobs)
        else:
            spmonte = spsyn.prepare(spobs)
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
        return out.flux

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

        # logg relation
        #  add a dummy logg value in
        if self.loggrelation:
            fullargs = np.insert(args,self.loggind,0.0)
        else:
            fullargs = args

        if self.verbose:
            print('jac: ',args)

        # Initialize jacobian matrix
        npix = len(wave)
        fjac = np.zeros((npix,self.nlabels),np.float64)
        
        # Loop over parameters
        pars = np.array(copy.deepcopy(args))
        f0 = self.model(wave,*pars,**kwargs)        
        steps = np.zeros(self.nlabels)
        for i in range(self.nlabels):
            if self.loggrelation and i==self.loggind:
                continue
            targs = np.array(copy.deepcopy(fullargs))
            if i==0:
                step = 10.0                
            else:
                step = 0.01
            steps[i] = step
            # Check boundaries, if above upper boundary
            #   go the opposite way
            if targs[i]>self.ranges[i,1]:
                step *= -1
            targs[i] += step
            # Remove dummy logg if using logg relation
            if self.loggrelation:
                targs = np.delete(targs,self.loggind)
            f1 = self.model(wave,*targs,**kwargs)
            fjac[:,i] = (f1-f0)/steps[i]
            
        self.njac += 1
            
        return fjac
        
    
    def fit(self,spec,fitparams=None,loggrelation=False,initgrid=True,
            outlier=False,verbose=False):
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
        ## Now normalize
        #newflux = fr.normalize(pspec['wave'],pspec['flux'])
        #cont = pspec['flux']/newflux
        #pspec['oflux'] = pspec['flux']
        #pspec['oerr'] = pspec['err']        
        #pspec['flux'] = newflux
        #pspec['err'] = pspec['oerr']/cont      
        ## Normalize
        #fr.outwave = pspec['wave']
        #if loggrelation:
        #    estimates = [4500.0,-1.0,0.0]
        #    bounds = [np.delete(fr.ranges[:,0],1),np.delete(fr.ranges[:,1],1)]            
        #else:
        #    estimates = [4500.0,2.5,-1.0,0.0]
        #    bounds = [fr.ranges[:,0],fr.ranges[:,1]]
        estimates = [4500.0,2.0,0.0,0.0]
        bounds = [self.ranges[:,0],self.ranges[:,1]]
        

        # Run set of ~100 points to get first estimate
        if initgrid:
            
            if loggrelation:
                nsample = 5
                tstep = np.ptp(self._ranges[:,0,:])/nsample
                tgrid = np.arange(nsample)*tstep+self._ranges[0,0,0]+tstep*0.5
                mstep = np.ptp(self._ranges[:,2,:])/nsample
                mgrid = np.arange(nsample)*mstep+self._ranges[0,2,0]+mstep*0.5
                astep = np.ptp(self._ranges[:,3,:])/nsample
                agrid = np.arange(nsample)*astep+self._ranges[0,3,0]+astep*0.5
                tgrid2d,mgrid2d,agrid2d = np.meshgrid(tgrid,mgrid,agrid)
                gridpars = np.vstack((tgrid2d.flatten(),mgrid2d.flatten(),agrid2d.flatten())).T
            else:
                nsample = 4
                tstep = np.ptp(self._ranges[:,0,:])/nsample/1.1
                tgrid = np.arange(nsample)*tstep+self._ranges[0,0,0]+tstep*0.5
                gstep = np.ptp(self._ranges[:,1,:])/nsample/1.1
                ggrid = np.arange(nsample)*gstep+self._ranges[0,1,0]+gstep*0.5            
                mstep = np.ptp(self._ranges[:,2,:])/nsample/1.1
                mgrid = np.arange(nsample)*mstep+self._ranges[0,2,0]+mstep*0.5
                astep = np.ptp(self._ranges[:,3,:])/nsample/1.1
                agrid = np.arange(nsample)*astep+self._ranges[0,3,0]+astep*0.5
                tgrid2d,ggrid2d,mgrid2d,agrid2d = np.meshgrid(tgrid,ggrid,mgrid,agrid)
                gridpars = np.vstack((tgrid2d.flatten(),ggrid2d.flatten(),mgrid2d.flatten(),agrid2d.flatten())).T
            if verbose:
               print('Testing an initial grid of '+str(gridpars.shape[0])+' spectra')
            
            # Make the models
            for i in range(gridpars.shape[0]):
                tpars1 = {'teff':gridpars[i,0],'logg':gridpars[i,1],
                          'mh':gridpars[i,2],'alpham':gridpars[i,3]}
                sp1 = self(tpars1)
                if i==0:
                    synflux = np.zeros((gridpars.shape[0],sp1.size),float)
                synflux[i,:] = sp1.flux
            chisqarr = np.sum((synflux-spec.flux)**2/spec.err**2,axis=1)/spec.size
            bestind = np.argmin(chisqarr)
            estimates = gridpars[bestind,:]
            estlabels = self.mklabels({'teff':estimates[0],'logg':estimates[1],
                                       'mh':estimates[2],'alpham':estimates[3]})
        else:
            estlabels = np.zeros(self.nlabels)
            estlabels[0:2] = [4200.0,1.5]
            
        if verbose:
            print('Initial estimates: ',estimates)
            
        try:
            pars,pcov = curve_fit(self.model,spec.wave,spec.flux,p0=estlabels,
                                  sigma=spec.err,bounds=bounds,jac=self.jac)
            perror = np.sqrt(np.diag(pcov))
            bestmodel = self.model(spec.wave,*pars)
            chisq = np.sum((spec.flux-bestmodel)**2/spec.err**2)/spec.size

            import pdb; pdb.set_trace()
            
            # Get full parameters
            if loggrelation:
                fullpars = self.getlogg(pars)
                fullperror = np.insert(perror,self.loggind,0.0)
            else:
                fullpars = pars
                fullperror = perror

            if verbose:
                printvals = [fullpars[0],fullperror[0],fullpars[1],fullperror[1],fullpars[2],fullperror[2],
                             fullpars[3],fullperror[3]]
                print('Best parameters: {:f}+/-{:.3g}, {:.3f}+/-{:.3g}, {:.3f}+/-{:.3g}, {:.3f}+/-{:.3g}'.format(*printvals))
                print('Chisq: ',chisq)

            # Construct the output dictionary
            out1 = {'vrel':vrel,'snr':spec.snr,'pars':fullpars,'perror':fullperror,'wave':spec.wave,
                    'flux':spec.flux,'err':spec.err,'model':bestmodel,'chisq':chisq,
                    'loggrelation':loggrelation,'success':True}
            success = True
        except:
            traceback.print_exc()
            success = False
            out1 = {'success':False}

        import pdb; pdb.set_trace()
                
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



            
    
def monte():
    """ Simple Monte Carlo test to recover elemental abundances."""


    
    
    

def test():

    # Simulate fake JWST data with ANN model
    em = Emulator.read('/Users/nidever/synspec/nodegrid/grid8/grid8_annmodel_300neurons_0.0001rate_20000steps.pkl')
    npix_syn = 22001
    wsyn = np.arange(npix_syn)*0.5+9000

    # need to convolve with the JWST LSF

    wobs_coef = np.array([-1.51930967e-09, -5.46761333e-06,  2.39684716e+00,  8.99994494e+03])
    # 3847 pixels
    npix_obs = 3847
    wobs = np.polyval(wobs_coef,np.arange(npix_obs))
    
    spobs = Spec1D(np.zeros(npix_obs),wave=wobs,err=np.ones(npix_obs),
                   lsfpars=np.array([ 1.05094118e+00, -3.37514635e-06]),
                   lsftype='Gaussian',lsfxtype='wave')

    pars = np.array([4000.0,2.0,0.0])
    pars = np.concatenate((pars,np.zeros(25)))
    spsyn = Spec1D(em(pars),wave=wsyn,err=np.ones(npix_syn))
    spmonte = spsyn.prepare(spobs)

    # deal with any NaNs
    bd, = np.where(~np.isfinite(spmonte.flux))
    if len(bd)>0:
        spmonte.flux[bd] = 1.0
        spmonte.err[bd] = 1e30
        spmonte.mask[bd] = True

        
