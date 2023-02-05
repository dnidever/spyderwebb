# Fit multi-band photometry with isochrones models to
# constrain Teff, A(V)

# Photometry Fitter (photter)

import numpy as np
from dlnpyutils import utils as dln
from scipy.optimize import curve_fit
from astropy.table import Table
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
    for i in range(ndata):
        out = fitone(data[i],iso,bands,refband)

    import pdb; pdb.set_trace()
    

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
    
    # Loop over metallicity and find the best solution
    for m in range(nmetals):
        metal = index['value'][m]
        ind = index['index'][index['lo'][m]:index['hi'][m]+1]
        nind = len(ind)
        iso1 = iso[ind]

        #xdata = [275,336,475,814,1100,1600]
        #phdata = np.array([data['f275w'],data['f336mag'],data['f475w'],data['f814w'],data['f110w'],data['f160w']])
        #ydata = phdata - phdata[3]
        xdata = np.zeros(len(bands),float)
        ydata = np.zeros(len(bands),float)
        refphot = data[refband]
        for i,b in enumerate(bands):
            phdata[i] = data[b]
        ydata = phdata - refphot
            
        #colors = np.zeros((len(iso),5),float)
        #colors[:,0] = iso['F275Wmag']-iso['F814Wmag']
        #colors[:,1] = iso['F336Wmag']-iso['F814Wmag']
        #colors[:,2] = iso['F475Wmag']-iso['F814Wmag']
        #colors[:,3] = iso['F110Wmag']-iso['F814Wmag']
        #colors[:,4] = iso['F160Wmag']-iso['F814Wmag']

        # Find best teff, AV, and log(age)
        estimates = [4500.0,0.1,np.median(iso['logAge'])]
        errors = np.zeros(len(xdata),float)+0.05
        bad, = np.where((phdata < 2) | (phdata>50) | (np.isfinite(phdata)==False))
        if len(bad)>0:
            ydata[bad] = 0.0
            errors[bad] = 1e30

        #bands = ['F275Wmag','F336Wmag','F475Wmag','F814Wmag','F110Wmag','F160Wmag']
        #refband = 'F814Wmag'
        fitter = Fitter(iso,bands,refband)
            
        pars,pcov = curve_fit(fitter.model,xdata,ydata,p0=estimates,sigma=errors)

        import pdb; pdb.set_trace()
