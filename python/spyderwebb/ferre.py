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


def interp(pars,wave=None,cont=None,ncont=None):
    """ Interpolate in the FERRE grid."""

    gridfile = '/Users/nidever/synspec/winter2017/jwst/jwstgiant2b.dat'
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
    

