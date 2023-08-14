import os
import numpy as np
from glob import glob
from dlnpyutils import utils as dln,coords
import doppler
from doppler import spec1d
from astropy.table import Table,hstack,Column
from astropy.io import fits
from chronos import isochrone
import shutil
import subprocess
import tempfile
import traceback
import matplotlib
import matplotlib.pyplot as plt
from . import utils

cspeed = 2.99792458e5  # speed of light in km/s

def continuum(sp,order=1):
    return spec1d.continuum(sp,norder=2,perclevel=50.0,binsize=0.2,interp=True)
    #x = np.linspace(-1,1,len(sp.wave))
    #good1 = (~sp.mask)
    #coef1,_ = dln.ladfit(x[good1],sp.flux[good1])
    #coef1 = coef1[::-1]
    #cont1 = np.polyval(coef1,x)    
    #sig = dln.mad(sp.flux/cont1)
    #good = ((np.abs(sp.flux/cont1-1) < 3*sig) & ~sp.mask)
    #if order==1:
    #    coef,_ = dln.ladfit(x[good],sp.flux[good])
    #    coef = coef[::-1]
    #else:   
    #    coef = robust.polyfit(x[good],sp.flux[good],order)
    #cont = np.polyval(coef,x)
    #return cont,coef

def readspec(filename):
    spec = doppler.read(filename)
    # Doppler models only go up to 18000
    wthresh = 17900
    bad = (spec.wave >= wthresh)
    if np.sum(bad)>0:
        # Trim to wave < 18000
        if spec.ndim==1:
            orig = spec.copy()
            gd, = np.where((spec.wave < wthresh) & (spec.wave > 0))
            npix = len(gd)
            spec.flux = np.zeros(npix,float)
            spec.err = np.zeros(npix,float)+1e30
            spec.mask = np.ones(npix,bool)
            spec.wave = np.zeros(npix,float)
            spec.flux = orig.flux[gd]
            spec.err = orig.err[gd]
            spec.wave = orig.wave[gd]
            spec.mask = orig.mask[gd]
            spec.numpix[0] = npix
            if hasattr(spec,'_cont') and spec._cont is not None:
                spec._cont = spec._cont[gd]
        else:
            orig = spec.copy()
            gd0, = np.where((spec.wave[:,0] < wthresh) & (spec.wave[:,0]>0))
            gd1, = np.where((spec.wave[:,1] < wthresh) & (spec.wave[:,1]>0))
            npix = max([len(gd0),len(gd1)])
            spec.flux = np.zeros([npix,2],float)
            spec.err = np.zeros([npix,2],float)+1e30
            spec.mask = np.ones([npix,2],bool)
            spec.wave = np.zeros([npix,2],float)
            if hasattr(spec,'_cont') and spec._cont is not None:
                spec.cont = np.zeros([npix,2],float)
            for i in range(2):
                gd, = np.where((orig.wave[:,i] < wthresh) & (orig.wave[:,i]>0))
                spec.numpix[i] = len(gd)
                spec.flux[0:len(gd),i] = orig.flux[gd,i]
                spec.err[0:len(gd),i] = orig.err[gd,i]
                spec.mask[0:len(gd),i] = orig.mask[gd,i]
                spec.wave[0:len(gd),i] = orig.wave[gd,i]                
                if hasattr(spec,'_cont') and spec._cont is not None:
                    spec._cont[0:len(gd),i] = spec._cont[gd,i]            
    spec._cont = continuum(spec)
    return spec


def doppler_joint(visitfiles,payne=False,clobber=False,verbose=True):
    """ Run Doppler jointly on visit files for a star."""

    visitfile = visitfiles[0]
    
    if payne:
        figfile = visitfile.replace('.fits','_doppler_joint_payne.png')
        outfile = visitfile.replace('.fits','_doppler_joint_payne.fits')
    else:
        figfile = visitfile.replace('.fits','_doppler_joint.png')
        outfile = visitfile.replace('.fits','_doppler_joint.fits')
    if os.path.exists(outfile) and clobber==False:
        print(outfile+' already exists')
        out = Table.read(outfile,1)
        return out
    if os.path.exists(outfile): os.remove(outfile)        

    # Load the files
    spec = []
    for i in range(len(visitfiles)):
        sp = readspec(visitfiles[i])
        spec.append(sp)
    try:
        out = doppler.jointfit(spec,verbose=verbose,payne=payne)
        sumstr, final, model, specmlist = out
        
        # Save the output
        if os.path.exists(outfile): os.remove(outfile)
        # Summary average values
        Table(sumstr).write(outfile)
        # append other items
        hdulist = fits.open(outfile)
        # append final values for each spectrum
        hdu = fits.table_to_hdu(Table(final))
        hdulist.append(hdu)
        # append best model
        # there's a model for each spectrum, each one gets an extension
        for i,m in enumerate(model):
            hdu = fits.PrimaryHDU(m.flux)
            hdulist.append(hdu)
        hdulist.writeto(outfile,overwrite=True)
        hdulist.close()
        
    except:
        out = None        
        traceback.print_exc()
        
    return out


def doppler_visit(visitfile,payne=False,estimates=None,verbose=True,clobber=False):
    """ Run Doppler one visit stellar spectrum."""
    if payne:
        figfile = visitfile.replace('.fits','_doppler_payne.png')
        outfile = visitfile.replace('.fits','_doppler_payne.fits')
    else:
        figfile = visitfile.replace('.fits','_doppler.png')
        outfile = visitfile.replace('.fits','_doppler.fits')
    if os.path.exists(visitfile) and os.path.getsize(visitfile)==0:
        print(visitfile+' is empty')
        return None,None,None
    if os.path.exists(outfile) and clobber==False:
        print(outfile+' already exists')
        out = Table.read(outfile,1)
        return out
    if os.path.exists(outfile): os.remove(outfile)        

    spec = readspec(visitfile)
    try:
        out,model,specm = doppler.fit(spec,estimates=estimates,verbose=verbose,
                                      figfile=figfile,payne=payne)
        out = Table(out)
        out['snr'] = spec.snr        
        # Save the output to a file
        Table(out).write(outfile)
        # append best model
        hdulist = fits.open(outfile)
        hdu = fits.PrimaryHDU(model.flux)
        hdulist.append(hdu)
        hdulist.writeto(outfile,overwrite=True)
        hdulist.close()
    except:
        out = None
        traceback.print_exc()
        
    return out


def doppler_stack(stackfile,payne=False,estimates=None,verbose=True,clobber=False):
    """ Run Doppler on the stacked stellar spectrum."""
    if payne:
        figfile = stackfile.replace('.fits','_doppler_payne.png')
        outfile = stackfile.replace('.fits','_doppler_payne.fits')
    else:
        figfile = stackfile.replace('.fits','_doppler.png')
        outfile = stackfile.replace('.fits','_doppler.fits')        
    if os.path.exists(outfile) and clobber==False:
        print(outfile+' already exists')
        out = Table.read(outfile,1)
        return out
    if os.path.exists(outfile): os.remove(outfile)  

    if os.path.getsize(stackfile)==0:
        print(stackfile+' is empty')
        return None    
    spec = readspec(stackfile)
    try:
        out,model,specm = doppler.fit(spec,estimates=estimates,verbose=verbose,
                                      figfile=figfile,payne=payne)
        out = Table(out)
        out['snr'] = spec.snr
        # Save the output to a file
        Table(out).write(outfile)
        # append best model
        hdulist = fits.open(outfile)
        hdu = fits.PrimaryHDU(model.flux)
        hdulist.append(hdu)
        hdulist.writeto(outfile,overwrite=True)
        hdulist.close()

    except:
        out = None
        traceback.print_exc()
        
    return out

def run_doppler(obsid,redtag='red',targfile=None,photfile=None,clobber=False,payne=False):
    """ Run Doppler on all the stars."""
    # obsid 'G140H-F100LP-M71_test68'
    
    print('Running doppler for '+obsid)
    
    # Get visit and stack files
    visitfiles = np.char.array(glob(obsid+'/spVisit-*'+redtag+'.fits'))
    visitfiles = np.char.array([v for v in visitfiles if os.path.getsize(v)>0])
    visitbases = [os.path.basename(v) for v in visitfiles if os.path.getsize(v)>0]
    vstarids = [v[8:] for v in visitbases]
    vstarids = [v[:v.find('_jw')] for v in vstarids]
    vstarids = np.char.array(vstarids)
    nvisits = len(visitfiles)

    # Get the list of Stacked files    
    stackfiles = np.char.array(glob(obsid+'/stack/spStack-*_'+redtag+'.fits*'))
    starids = [os.path.basename(c)[8:-len(redtag)-6] for c in stackfiles if os.path.getsize(c)>0]
    starids = np.char.array(starids)
    nstars = len(starids)
    print(nstars,' stars')

    # Loop over the stars
    slist = {}
    vlist = {}
    for i in range(nstars):
        starid = starids[i]
        print(i+1,starid)
        # Visit files
        vind, = np.where(vstarids==starid)
        vfiles = visitfiles[vind]
        #vout = doppler_visit(vfiles)
        # Loop over visit files
        for j in range(len(vfiles)):
            vout = doppler_visit(vfiles[j],clobber=clobber)
            vlist[vfiles[j]]= vout
            if payne:
                estimates = {'TEFF':vout['teff'][0],'LOGG':vout['logg'][0],'FE_H':vout['feh'][0],'RV':vout['vrel'][0]}
                vout_payne = doppler_visit(vfiles[j],clobber=clobber,payne=True,estimates=estimates)
                
        # Run doppler with joint on visits
        if len(vind)>1:
            jout = doppler_joint(vfiles,clobber=clobber)
        
        # Stack file
        sind, = np.where(starids==starid)
        sfiles = stackfiles[sind[0]]
        sout = doppler_stack(sfiles,clobber=clobber)
        slist[starid] = sout
        if payne:
            estimates = {'TEFF':sout['teff'][0],'LOGG':sout['logg'][0],'FE_H':sout['feh'][0],'RV':sout['vrel'][0]}
            sout_payne = doppler_stack(sfiles,clobber=clobber,payne=True,estimates=estimates)        

    # Create visit catalog
    dt = [('starid',str,50),('visitfile',str,200),('id',str,50),('vhelio',float),('vrel',float),('vrelerr',float),
          ('teff',float),('tefferr',float),('logg',float),('loggerr',float),('feh',float),('feherr',float),
          ('chisq',float),('bc',float),('snr',float),('success',bool)]
    vtab = np.zeros(len(vlist),dtype=np.dtype(dt))
    for i,k in enumerate(vlist.keys()):
        vout = vlist[k]
        # G140H-F100LP-M31-FINAL-LONG/spVisit-2609_268389_jw02609009001_04107_00001_red.fits
        vfile = os.path.basename(k)
        starid = k[8:]
        starid = '_'.join(starid.split('_')[0:2])
        vtab['starid'][i] = starid
        vtab['id'][i] = starid.split('_')[1]            
        vtab['visitfile'][i] = vfile
        if vout is None:
            vtab['success'][i] = False
        else:
            for n in vout.dtype.names:
                if n in vout.columns:
                    vtab[n][i] = vout[n][0]
            vtab['success'][i] = True
    vtab = Table(vtab)
            
    # Create stack catalog
    dt = [('starid',str,50),('id',str,50),('vhelio',float),('vrel',float),('vrelerr',float),('teff',float),('tefferr',float),
          ('logg',float),('loggerr',float),('feh',float),('feherr',float),('chisq',float),('bc',float),('snr',float),('success',bool)]
    tab = np.zeros(len(slist),dtype=np.dtype(dt))
    for i,k in enumerate(slist.keys()):
        out = slist[k]
        if out is None:
            tab['starid'][i] = k
            tab['success'][i] = False
        else:
            for n in out.dtype.names:
                if n in out.columns:
                    tab[n][i] = out[n][0]
            tab['starid'][i] = k
            tab['id'][i] = k.split('_')[1]
            tab['success'][i] = True
    tab = Table(tab)


    # Add targeting information
    if targfile is not None:
        targs = Table.read(targfile)
        for c in targs.colnames: targs[c].name = c.lower()
        if 'id' not in targs.columns:
            targs['id'] = np.arange(len(targs))+2  # APT has first ID as 2

        ind1,ind2 = dln.match(tab['id'],targs['id'])
        print(len(ind1),' matches to targeting catalog')
        # Add the new columns
        if len(ind1)>0:        
            for k in targs.columns:
                if k not in tab.columns:
                    tab.add_column(Column(name=k,dtype=targs[k].dtype,length=len(tab)))
                    tab[k][ind1] = targs[k][ind2]

    # Add other catalog information
    if photfile is not None:
        # Match up with catalog that has photometry and other information
        phot = Table.read(photfile)
        for c in phot.colnames: phot[c].name = c.lower()

        ind1,ind2,dist = coords.xmatch(tab['ra'],tab['dec'],phot['ra'],phot['dec'],0.5)
        print(len(ind1),' matches to photometry catalog')
        # Add the new columns
        if len(ind1)>0:        
            for k in phot.columns:
                if k not in tab.columns:
                    tab.add_column(Column(name=k,dtype=phot[k].dtype,length=len(tab)))
                    tab[k][ind1] = phot[k][ind2]
                    
    return tab,vtab


def run_ferre(files,vrel,inter=3,algor=1,init=1,indini=None,nruns=1,cont=1,ncont=0,
              errbar=1,grid='jwstgiant4.dat',save=False,plotsdir=None,plots=True):
    """ 
    Run FERRE on list of spectra

    Parameters
    ----------
    files : list
       Input list of spectrum files.
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

    tab,info = run_ferre(files,vrel,algor=1,nruns=5)

    """

    #if type(files) != list:
    #    files = [files]
    #if type(vrel) != list:
    #    vrel = [vrel]
    nfiles = len(files)

    # Loop over the files
    slist = []
    for i in range(nfiles):
        filename = files[i]
        vrel1 = vrel[i]
        if filename is None:
            continue
        if os.path.exists(filename)==False:
            print(filename+' not found')
            continue
        if os.path.getsize(filename)==0:
            print(filename+' is empty')
            continue
        spec = doppler.read(filename)
        spec.vrel = vrel1
        #spec.normalize()
        slist1 = {'filename':filename,'vrel':vrel1,'snr':spec.snr,'spec':spec}
        slist.append(slist1)

    if len(slist)==0:
        print('No spectra to fit')
        return None,None
        
    gridfile = '/Users/nidever/synspec/winter2017/jwst/'+grid
    ferre = '/Users/nidever/projects/ferre/bin/ferre.x'

    # Set up temporary directory
    tmpdir = tempfile.mkdtemp(prefix='ferre')
    curdir = os.path.abspath(os.curdir)
    if plots and plotsdir is None:
        plotsdir = curdir+'/plots/'
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

    # Prepare the spectra for input to FERRE
    for i in range(len(slist)):
        slist1 = slist[i]
        filename = slist1['filename']
        spec = slist1['spec']        
        print(i+1,filename,spec.vrel)
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
        wave /= (1+spec.vrel/cspeed)
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
        slist1['npix'] = len(flux)
        slist1['flux'] = flux
        slist1['err'] = err
        slist1['wave'] = wave
        
    # Put star with maximum npix first
    #  FERRE uses this to set the maximum pixels for ALL the spectra
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

    # Sometimes the number of lines in opf and mdl/nrd do not match

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
        try:
            snr2 = 10**logsnr2            
            logrchisq = float(arr[11])
            rchisq = 10**logrchisq
        except:
            snr2 = np.nan
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

        # Make plots
        if plots:
            if os.path.exists(plotsdir)==False: os.makedirs(plotsdir)
            backend = matplotlib.rcParams['backend']
            matplotlib.use('Agg')
            figsize = 10.0
            fig,ax = plt.subplots()
            fig.set_figheight(figsize*0.5)
            fig.set_figwidth(figsize)
            plt.plot(slist1['wave'],slist1['smflux'],linewidth=0.5)
            plt.plot(slist1['wave'],slist1['model'],linewidth=0.5,alpha=0.8)
            yr = [np.min(slist1['model']),np.max(slist1['model'])]
            yr = [yr[0]-np.ptp(yr)*0.2,yr[1]+np.ptp(yr)*0.2]
            medflux = np.median(slist1['smflux'])
            sigflux = dln.mad(slist1['smflux'])
            yr = [np.min([yr[0],medflux-2.5*sigflux]),np.max([yr[1],medflux+2.5*sigflux])]
            plt.ylim(yr[0],yr[1])
            plt.xlim(np.min(slist1['wave']),np.max(slist1['wave']))
            plt.xlabel('Wavelength (A)')
            plt.ylabel('Normalized Flux')
            plt.title(slist1['filename'])
            plotfile = plotsdir+'/'+os.path.basename(slist1['filename']).replace('.fits','_ferre.pdf')
            plt.savefig(plotfile)
            print('Saving plot to ',plotfile)
            matplotlib.use(backend)  # back to the original backend
            
    # Make a table
    dt = [('filename',str,200),('vrel',float),('snr',float),('teff',float),('tefferr',float),
          ('logg',float),('loggerr',float),('feh',float),('feherr',float),('alpha',float),
          ('alphaerr',float),('pars',float,4),('parerr',float,4),('snr2',float),('rchisq',float)]
    tab = np.zeros(len(slist),dtype=np.dtype(dt))
    for i in range(len(slist)):
        slist1 = slist[i]
        tab['filename'][i] = slist1['filename']
        tab['vrel'][i] = slist1['vrel']
        tab['snr'][i] = slist1['snr']
        if 'pars' in slist1.keys() and slist1['pars'] is not None:
            tab['teff'][i] = slist1['pars'][0]
            tab['tefferr'][i] = slist1['parerr'][0]
            tab['logg'][i] = slist1['pars'][1]
            tab['loggerr'][i] = slist1['parerr'][1]
            tab['feh'][i] = slist1['pars'][2]
            tab['feherr'][i] = slist1['parerr'][2]
            tab['alpha'][i] = slist1['pars'][3]
            tab['alphaerr'][i] = slist1['parerr'][3]
            tab['pars'][i] = slist1['pars']
            tab['parerr'][i] = slist1['parerr']
            tab['snr2'][i] = slist1['snr2']
            tab['rchisq'][i] = slist1['rchisq']
    tab = Table(tab)
    
    # Delete temporary files and directory
    if save==False:    
        os.chdir(curdir)
        shutil.rmtree(tmpdir)

    return tab,slist

def distances(tab,isogrid=None):
    """
    Derives distances from isochrones.
    """

    if isogrid is None:
        print('Loading isochrone grid')
        isotab = Table.read('/Users/nidever/isochrone/parsec_hst/parsec_hst_phot_giants.fits.gz')
        for c in isotab.colnames: isotab[c].name=c.upper()
        isotab['AGE'] = 10**isotab['LOGAGE']
        isotab['METAL'] = isotab['MH']
        isotab['F275WMAG'].name = 'HST_F275WMAG'
        isotab['F336WMAG'].name = 'HST_F336WMAG'
        isotab['F475WMAG'].name = 'HST_F475WMAG'
        isotab['F814WMAG'].name = 'HST_F814WMAG'
        isotab['F110WMAG'].name = 'HST_F110WMAG'
        isotab['F160WMAG'].name = 'HST_F160WMAG'
        isogrid = isochrone.IsoGrid(isotab)
        
    # Each star must have Teff, logg, feh and alpha
    # We'll get distances for a couple ages

    ntab = len(tab)
    dt = [('dist1',float),('dist3',float),('dist5',float),('dist7',float),('dist10',float),('dist12',float),('mndist',float)]
    out = np.zeros(ntab,dtype=np.dtype(dt))
    ages = [1e9, 3e9, 5e9, 7e9, 10e9, 12e9]
    # Loop over the stars
    for i in range(ntab):
        teff = tab['teff'][i]
        logg = tab['logg'][i]
        feh = tab['feh'][i]
        alpha = tab['alpha'][i]        
        # Salaris correction feh
        salfeh = feh + np.log10(0.659*(10**alpha)+0.341)
        usefeh = np.maximum(np.minimum(salfeh,isogrid.maxmetal),isogrid.minmetal)
        # Age loop
        distarr = np.zeros(6)
        for a,age in enumerate(ages):
            # Get the isochrone
            iso = isogrid(age,usefeh,closest=True)

            # EXTINCTION from BEAST
            av = tab['beast_Av'][i]
            rv = tab['beast_Rv'][i]
            iso.ext = av   # extinct the isochrone
            
            data1 = iso.data[0:1]
            for c in iso.colnames:
                data1[c] = dln.interp(iso.data['LOGTE'],iso.data[c],np.log10(teff),kind='quadratic',assume_sorted=False)
            
            # Get distance modulus from f160w and f475w
            agename = 'dist'+str(int(age/1e9))
            obsmags = [tab['f275w'][i], tab['f336w'][i], tab['f475w'][i], tab['f814w'][i], tab['f110w'][i], tab['f160w'][i]]
            obsmags = np.array(obsmags)
            isomags = [data1['HST_F275WMAG'][0], data1['HST_F336WMAG'][0], data1['HST_F475WMAG'][0], data1['HST_F814WMAG'][0],
                       data1['HST_F110WMAG'][0],data1['HST_F160WMAG'][0]]
            isomags = np.array(isomags)
            distmodarr = obsmags - isomags
            good, = np.where((obsmags < 50) & (isomags < 0))
            if len(good)>0:
            #if data1['HST_F160WMAG'][0] < 0:
                #distmodarr = [tab['f275w'][i]-data1['HST_F275WMAG'][0],
                #              tab['f336w'][i]-data1['HST_F336WMAG'][0],
                #              tab['f475w'][i]-data1['HST_F475WMAG'][0],
                #              tab['f814w'][i]-data1['HST_F814WMAG'][0],
                #              tab['f110w'][i]-data1['HST_F110WMAG'][0],
                #              tab['f160w'][i]-data1['HST_F160WMAG'][0]]
                distmod = np.mean(distmodarr[good])
                dist = 10**(1+distmod/5) / 1e3  # kpc
                out[agename][i] = dist
            else:
                out[agename][i] = np.nan
            distarr[a] = out[agename][i]
            
        mndist = np.nanmean(distarr)
        out['mndist'][i] = mndist
        
        fmt = '{:5d} {:9.1f} {:7.3f} {:8.3f} {:8.3f} {:8.3f}  {:9.1f} {:9.1f} {:9.1f} {:9.1f} {:9.1f} {:9.1f}  {:9.1f}'
        print(fmt.format(i+1,teff,logg,feh,alpha,usefeh,*distarr,mndist))
        
    return out
