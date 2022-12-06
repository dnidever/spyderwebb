import os
import numpy as np
from dlnpyutils import utils as dln
from astropy.table import Table
from glob import glob



def qa(obsid):
    """ Make QA html pages."""

    outdir = obsid+'/qa/'
    if os.path.exists(outdir)==False:
        os.makedirs(outdir)
    
    lines = []
    lines += ['<html>']
    lines += ['<head>']
    lines += ['<title>Observation '+obsid]
    lines += ['</title>']
    lines += ['</head>']
    lines += ['<body>']        


    lines += ['<h1>Observation '+obsid+'</h1>']
    lines += ['<p>']

    # Get visit and stack files
    visitfiles = glob(obsid+'/spVisit-*fits*')
    visitfiles = [os.path.basename(v) for v in visitfiles if os.path.getsize(v)>0]
    vstarids = [v[8:] for v in visitfiles]
    vstarids = [v[:v.find('_jw')] for v in vstarids]
    nvisits = len(visitfiles)

    # Get exposure names
    expnames = [os.path.basename(v) for v in visitfiles]
    expnames = np.char.array([e[e.find('_jw')+1:] for e in expnames])
    expnames = expnames.replace('_cal.fits','')
    expnames = expnames.replace('_rate.fits','')
    expnames = np.unique(expnames)
    nexp = len(expnames)
    
    calstackfiles = glob(obsid+'/stack/spStack-*_cal.fits*')
    starids = [os.path.basename(c)[8:-9] for c in calstackfiles if os.path.getsize(c)>0]
    nstars = len(starids)
    
    lines += ['<table border=1>']
    header = '<tr><th>Number</th><th>Name</th><th>Image Type</th>'
    for e in range(nexp):
        exp = expnames[e]
        header += '<th>'+exp+' NRS1 Image</th><th>'+exp+' NRS2 Image</th>'
        header += '<th>'+exp+' NRS1 PSF</th><th>'+exp+' NRS2 PSF</th>'
        header += '<th>'+exp+' NRS1 Flux</th><th>'+exp+' NRS2 Flux</th>'
    header += '<td>Combined Flux</th>'
    lines += [header]

    # A separate row for each star
    for i in range(nstars):
        starid = starids[i]
        lines += ['<tr>']
        lines += ['<td rowspan=2>'+str(i+1)+'</td>']
        lines += ['<td rowspan=2>'+starid+'</td>']
        lines += ['<td>Cal</td>']
        # Cal plots
        for e in range(nexp):
            exp = expnames[e]
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs1_slitdata.png"><img src="../plots/'+starid+'_'+exp+'_nrs1_slitdata.png" height=400></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs2_slitdata.png"><img src="../plots/'+starid+'_'+exp+'_nrs2_slitdata.png" height=400></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs1_slitopsf.png"><img src="../plots/'+starid+'_'+exp+'_nrs1_slitopsf.png" height=400></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs2_slitopsf.png"><img src="../plots/'+starid+'_'+exp+'_nrs2_slitopsf.png" height=400></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs1_slitflux.png"><img src="../plots/'+starid+'_'+exp+'_nrs1_slitflux.png" height=400></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs2_slitflux.png"><img src="../plots/'+starid+'_'+exp+'_nrs2_slitflux.png" height=400></a></td>']                        
        lines += ['<td><img src="../stack/plots/spStack-'+starid+'_cal_flux.png" height=400></td>']
        #lines += ['<td><img src="../stack/plots/spStack-'+starid+'_rate_flux.png"></td>']        
        lines += ['<\tr>']
        
        # Rate plots
        lines += ['<tr>']
        lines += ['<td>Rate</td>']
        for e in range(nexp):
            exp = expnames[e]
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs1_ratedata.png"><img src="../plots/'+starid+'_'+exp+'_nrs1_ratedata.png" height=400></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs2_ratedata.png"><img src="../plots/'+starid+'_'+exp+'_nrs2_ratedata.png" height=400></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs1_rateopsf.png"><img src="../plots/'+starid+'_'+exp+'_nrs1_rateopsf.png" height=400></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs2_rateopsf.png"><img src="../plots/'+starid+'_'+exp+'_nrs2_rateopsf.png" height=400></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs1_rateflux.png"><img src="../plots/'+starid+'_'+exp+'_nrs1_rateflux.png" height=400></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs2_rateflux.png"><img src="../plots/'+starid+'_'+exp+'_nrs2_rateflux.png" height=400></a></td>']            
        lines += ['<td><img src="../stack/plots/spStack-'+starid+'_rate_flux.png" height=400></td>']
        #lines += ['<td><img src="../stack/plots/spStack-'+starid+'_rate_flux.png"></td>']        
        lines += ['<\tr>']
        
        # a row for Cal and another one for Rate
    
    
    lines += ['</table>']
    lines += ['</body>']     
    lines += ['</html>']    

    outfile = outdir+obsid+'_qa.html'
    print('Writing QA html file to ',outfile)
    dln.writelines(outfile,lines)
