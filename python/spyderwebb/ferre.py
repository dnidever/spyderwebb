import os
import numpy as np
import subprocess
import tempfile
import traceback
import shutil
from dlnpyutils import utils as dln

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


def interp(pars,wave=None,cont=None,ncont=None,grid='jwstgiant4.dat'):
    """ Interpolate in the FERRE grid."""

    gridfile = '/Users/nidever/synspec/winter2017/jwst/'+grid
    ferre = '/Users/nidever/projects/ferre/bin/ferre.x'
    info = gridinfo(gridfile)
    
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
    lines += ["NOBJ = 1"]
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
    flines = 'test1 {:.3f} {:.3f} {:.3f} {:.3f}'.format(*pars)
    dln.writelines('ferre.ipf',flines)
    
    # Run FERRE
    if os.path.exists('ferre.mdl'): os.remove('ferre.mdl')
    try:
        result = subprocess.check_output([ferre],stderr=subprocess.STDOUT)
    except:
        #traceback.print_exc()
        pass
    
    # Read the output
    mlines = dln.readlines('ferre.mdl')[0]
    mflux = np.array(mlines.split()).astype(float)
    
    # Load the wavelengths
    if wave is None:
        wave = info['WAVELENGTH'].copy()

    out = {'wave':wave,'flux':mflux}
        
    # Delete temporary files and directory
    os.chdir(curdir)
    shutil.rmtree(tmpdir)

    return out
    


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
