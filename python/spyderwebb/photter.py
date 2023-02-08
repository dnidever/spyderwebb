# Fit multi-band photometry with isochrones models to
# constrain Teff, A(V)

# Photometry Fitter (photter)

import numpy as np
from dlnpyutils import utils as dln
from scipy.optimize import curve_fit
from astropy.table import Table,hstack,vstack
import traceback
from . import extinction

class Fitter(object):

    # Fits stars with single metallicity isochrone
    
    def __init__(self,iso,bands,refband,extdict=None):
        self.iso = iso
        self.bands = bands
        self.nbands = len(bands)
        self.refband = refband
        metals = np.unique(iso['MH'].data)
        nmetals = len(metals)
        self.metals = metals
        self.nmetals = nmetals        
        ages = np.unique(iso['logAge'].data)
        nage = len(ages)
        self.logages = ages
        self.ages = 10**ages / 1e6   # in Myrs
        self.nages = len(ages)
        # Create an index of ages        
        if self.nages>0:
            age_index = dln.create_index(iso['logAge'].data)
            aindex = []
            for i in range(self.nages):
                aind = age_index['index'][age_index['lo'][i]:age_index['hi'][i]+1]
                aindex.append(aind)
            self.index = aindex
            self._age_index = age_index
        else:
            self.index = [np.arange(len(iso))]
        # Extinction dictionary
        if extdict is not None:
            self._extdict = extdict
        else:
            extdict,exttab = extinction.load()
            self._extdict = extdict
            self._exttab = exttab

    def __call__(self,*pars):
        """ Interpolate in the grid."""
        # Pars are teff, log(age)
        teff = pars[0]
        logteff = np.log10(teff)
        logage = pars[1]
        # Get model values

        phot = Table(np.zeros(1,dtype=self.iso.dtype))
        for c in phot.colnames: phot[c]=np.nan
        
        # Beyond ages in the model, return bad values
        if logage < np.min(self.logages) or logage >np.max(self.logages):
            return phot
            
        # At the edge
        if logage == np.min(self.logages) or logage == np.max(self.logages):
            if logage==np.min(self.logages):
                iso1 = self.iso[self.index[0]]
            else:
                iso1 = self.iso[self.index[-1]]                
            # Interpolate all iso columns
            for c in phot.colnames:
                phot[c] = dln.interp(iso1['logTe'],iso1[c],logteff,kind='quadratic',bounds_error=False,fill_value=np.nan)

        # We have two points
        else:
        
            # Get high and low ages
            aind = np.searchsorted(self.logages,logage)   # returns index just above the value
            alo = aind-1
            loage = self.logages[alo]
            loiso = self.iso[self.index[alo]]
            ahi = aind
            hiage = self.logages[ahi]
            hiiso = self.iso[self.index[ahi]]
            # wtfrac is weight of lower value, (1-wtfrac) is weight of upper value        
            wtfrac = (hiage-logage)/(hiage-loage)  

            # Interpolate all iso columns            
            for c in phot.colnames:            
                loval = dln.interp(loiso['logTe'],loiso[c],logteff,kind='quadratic',assume_sorted=False,
                                   bounds_error=False,fill_value=np.nan)
                hival = dln.interp(hiiso['logTe'],hiiso[c],logteff,kind='quadratic',assume_sorted=False,
                                   bounds_error=False,fill_value=np.nan)
                lofinite = np.isfinite(loval)
                hifinite = np.isfinite(hival)
                vals = []
                if lofinite: vals.append(loval)
                if hifinite: vals.append(hival)
                if len(vals)==0:
                    phot[c] = np.nan
                elif len(vals)==1:
                    phot[c] = vals[0]
                else:
                    phot[c] = wtfrac*vals[0] + (1-wtfrac)*vals[1]
                    
        return phot

        
    def model(self,x,*pars):
        # Pars are teff, A(V), log(age)
        teff = pars[0]
        logteff = np.log10(teff)
        av = pars[1]
        logage = pars[2]
        # Get model values

        # Beyond ages in the model, return bad values
        if logage < np.min(self.logages) or logage >np.max(self.logages):
            out = np.zeros(self.nbands)+1e30
            return out
            
        # At the edge
        if logage == np.min(self.logages) or logage == np.max(self.logages):
            if logage==np.min(self.logages):
                iso1 = self.iso[self.index[0]]
            else:
                iso1 = self.iso[self.index[-1]]                
            # Interpolate bands                
            phot = np.zeros(len(self.bands)+1,float)+99.99        
            for i,b in enumerate(self.bands+[self.refband]):
                phot[i] = dln.interp(iso1['logTe'],iso1[b],logteff,kind='quadratic',bounds_error=False,fill_value=np.nan)

        # We have two points
        else:
        
            # Get high and low ages
            aind = np.searchsorted(self.logages,logage)   # returns index just above the value
            alo = aind-1
            loage = self.logages[alo]
            loiso = self.iso[self.index[alo]]
            ahi = aind
            hiage = self.logages[ahi]
            hiiso = self.iso[self.index[ahi]]
            # wtfrac is weight of lower value, (1-wtfrac) is weight of upper value        
            wtfrac = (hiage-logage)/(hiage-loage)  
        
            lophot = np.zeros(len(self.bands)+1,float)+99.99
            hiphot = np.zeros(len(self.bands)+1,float)+99.99
            phot = np.zeros(len(self.bands)+1,float)+99.99        
            for i,b in enumerate(self.bands+[self.refband]):
                lophot[i] = dln.interp(loiso['logTe'],loiso[b],logteff,kind='quadratic',assume_sorted=False,
                                       bounds_error=False,fill_value=np.nan)
                hiphot[i] = dln.interp(hiiso['logTe'],hiiso[b],logteff,kind='quadratic',assume_sorted=False,
                                       bounds_error=False,fill_value=np.nan)
                lofinite = np.isfinite(lophot[i])
                hifinite = np.isfinite(hiphot[i])
                vals = []
                if lofinite: vals.append(lophot[i])
                if hifinite: vals.append(hiphot[i])
                if len(vals)==0:
                    phot[i] = np.nan
                elif len(vals)==1:
                    phot[i] = vals[0]
                else:
                    phot[i] = wtfrac*vals[0] + (1-wtfrac)*vals[1]
                    
        # Now apply the extinction
        data = Table()
        for i,b in enumerate(self.bands+[self.refband]):
            data[b] = [phot[i]]
        newdata = extinction.extinct(data,av,extdict=self._extdict,isonames=self.bands+[self.refband])
        
        # Subtract reference band
        colors = np.zeros(len(self.bands),float)
        refphot = newdata[self.refband]
        for i,b in enumerate(self.bands):
            colors[i] = newdata[b]-refphot
                    
        return colors

    
def fitter(data,iso,bands,refband):
    """
    data : photometry data
    iso : isochrone data
    """

    ndata = len(data)
    
    # Loop over stars
    results = []
    for i in range(ndata):
        print('--- Star {:d} ---'.format(i+1))
        besttab,sumtab = fitone(data[i],iso,bands,refband)
        print(' ')
        # Save the results
        if len(besttab)>0:
            besttab['index'] = i+1
            if len(results)==0:
                results = besttab
            else:
                results = vstack((results,besttab))

    return results


def fitone(data,iso,bands,refband):
    """
    data : photometry for single star
    iso : isochrone data
    """

    metals = np.unique(iso['MH'])
    nmetals = len(metals)
    ages = np.unique(iso['logAge'])
    nages = len(ages)

    index = dln.create_index(iso['MH'])

    dt = [('metal',float),('pars',float,3),('perrors',float,3),('chisq',float),('rms',float),('ngood',int),('success',bool)]
    sumtab = np.zeros(nmetals,dtype=np.dtype(dt))
    sumtab = Table(sumtab)

    xdata = np.zeros(len(bands),float)
    ydata = np.zeros(len(bands),float)
    errors = np.zeros(len(xdata),float)+0.05
    phdata = np.zeros(len(bands),float)
    refphot = data[refband]
    for i,b in enumerate(bands):
        phdata[i] = data[b]
    ydata = phdata - refphot
    good, = np.where((phdata > 2) & (phdata<50) & np.isfinite(phdata))
    ngood = len(good)
    if ngood==0:
        print('No good data to fit')
        return [],[]
    nbad = len(ydata)-len(good)
    if nbad>0:
        print(str(nbad)+' bad measurement(s)')
    bands1 = list(np.array(bands)[good])
    xdata = xdata[good]
    ydata = ydata[good]
    errors = errors[good]

    print('Bands: '+', '.join(bands1))
    print('Reference Band: '+str(refband))
    
    # Loop over metallicity and find the best solution
    lastpars = None
    print('  [M/H]    Teff      A(V)  log(Age)  Chisq     RMS')
    for m in range(nmetals):
        metal = index['value'][m]
        ind = index['index'][index['lo'][m]:index['hi'][m]+1]
        nind = len(ind)
        iso1 = iso[ind]

        fitter = Fitter(iso1,bands1,refband)        
        # Find best teff, AV, and log(age)
        if lastpars is None:
            estimates = [4500.0,0.1,np.median(iso['logAge'])]
        else:
            estimates = lastpars
        # physical bounds
        bounds = [np.zeros(3)-np.inf,np.zeros(3)+np.inf]
        bounds[0][0] = 10**np.min(iso['logTe'])
        bounds[1][0] = 10**np.max(iso['logTe'])        
        bounds[0][1] = 0.0
        bounds[0][2] = np.min(iso['logAge'])
        bounds[1][2] = np.max(iso['logAge'])
        try:
            pars,pcov = curve_fit(fitter.model,xdata,ydata,p0=estimates,sigma=errors,bounds=bounds)
            perrors = np.sqrt(np.diag(pcov))
            obscolors = fitter.model(xdata,*pars)
            chisq = np.sum((ydata-obscolors)**2/errors**2)
            rms = np.sqrt(np.mean((ydata-obscolors)**2))
            sumtab['metal'][m] = metal
            sumtab['pars'][m] = pars
            sumtab['perrors'][m] = perrors
            sumtab['chisq'][m] = chisq
            sumtab['rms'][m] = rms
            sumtab['ngood'][m] = ngood
            sumtab['success'][m] = True
            print('{:7.2f} {:9.3f} {:7.3f} {:7.3f} {:9.3f} {:7.3f}'.format(metal,*pars,chisq,rms))
            lastpars = pars
        except:
            traceback.print_exc()
            sumtab['metal'][m] = metal
            sumtab['ngood'][m] = ngood
            sumtab['chisq'][m] = np.nan
            sumtab['rms'][m] = np.nan            
            sumtab['success'][m] = False

    # Best value
    bestind = np.argmin(sumtab['chisq'])
    besttab = sumtab[[bestind]]
    besttab['teff'] = 0.0
    besttab['tefferr'] = 0.0    
    besttab['av'] = 0.0
    besttab['averr'] = 0.0    
    besttab['logage'] = 0.0
    besttab['logageerr'] = 0.0
    besttab['agemyr'] = 0.0    
    besttab['teff'] = besttab['pars'][0][0]
    besttab['tefferr'] = besttab['perrors'][0][0]    
    besttab['av'] = besttab['pars'][0][1]
    besttab['averr'] = besttab['perrors'][0][1]    
    besttab['logage'] = besttab['pars'][0][2]
    besttab['logageerr'] = besttab['perrors'][0][2]
    besttab['agemyr'] = (10**besttab['logage'][0])/1e6
    
    print('Best values:')
    print('{:7.2f} {:9.3f} {:7.3f} {:7.3f} {:9.3f} {:7.3f}'.format(besttab['metal'][0],*besttab['pars'][0],besttab['chisq'][0],besttab['rms'][0]))
    
    return besttab,sumtab

