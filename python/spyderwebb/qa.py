import os
import numpy as np
from dlnpyutils import utils as dln
from astropy.table import Table
from glob import glob



def qa(obsid,redtag='red'):
    """ Make QA html pages."""

    outdir = obsid+'/qa/'
    if os.path.exists(outdir)==False:
        os.makedirs(outdir)
    
    lines = []
    lines += ['<html>']
    lines += ['<head>']
    lines += ['<title>Observation '+obsid]
    lines += ['</title>']
    lines += ['<base target="_blank">']
    lines += ['</head>']
    lines += ['<body>']        


    lines += ['<h1>Observation '+obsid+'</h1>']
    lines += ['<p>']

    # Get visit and stack files
    visitfiles = glob(obsid+'/spVisit-*'+redtag+'*fits*')
    visitfiles = [os.path.basename(v) for v in visitfiles if os.path.getsize(v)>0]
    vstarids = [v[8:] for v in visitfiles]
    vstarids = [v[:v.find('_jw')] for v in vstarids]
    nvisits = len(visitfiles)
    starids = np.unique(vstarids)    
    nstars = len(starids)    

    # Get exposure names
    expnames = [os.path.basename(v) for v in visitfiles]
    expnames = np.char.array([e[e.find('_jw')+1:] for e in expnames])
    expnames = expnames.replace('_'+redtag+'.fits','')
    expnames = np.unique(expnames)
    nexp = len(expnames)
    lines += ['<h4>'+str(nexp)+' exposures: ']
    lines += ['<ul>']
    for i in range(nexp):
        lines += ['<li>'+expnames[i]]
    lines += ['</ul></h4>']
    
    calstackfiles = glob(obsid+'/stack/spStack-*_'+redtag+'.fits*')
    #starids = [os.path.basename(c)[8:-len(redtag)-5] for c in calstackfiles if os.path.getsize(c)>0]
    #nstars = len(starids)

    lines += ['<h3>Click on an image to see a larger version</h3><p>']
    
    lines += ['<table border=1>']
    header = '<tr bgcolor=#AEB6BF><th>Number</th><th>Name</th><th>Image Type</th>'
    colors = ['#85C1E9','#F5B041','#BB8FCE','#27AE60','#EC7063','#008080']   # blue,orange,purple,green,red,teal
    for e in range(nexp):
        exp = expnames[e]
        color = colors[e]
        header += '<th colspan=2 bgcolor='+color+'>Exposure '+str(e+1)+' Image</th>'
    header += '<td align=center>Combined Flux</th>'
    lines += [header]
    lines += ['<tr bgcolor=#AEB6BF><td></td><td></td><td></td>'+nexp*'<td align=center>NRS1</td><td align=center>NRS2</td>'+'<td></td></tr>']
    
    # A separate row for each star
    for i in range(nstars):
        starid = starids[i]
        lines += ['<tr>']
        lines += ['<td rowspan=3 align=center>'+str(i+1)+'</td>']
        lines += ['<td rowspan=3 align=center>'+starid+'</td>']
        lines += ['<td align=center>Image</td>']
        # Image plots
        for e in range(nexp):
            exp = expnames[e]
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs1_data.png"><img src="../plots/'+starid+'_'+exp+'_nrs1_data.png" height=200></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs2_data.png"><img src="../plots/'+starid+'_'+exp+'_nrs2_data.png" height=200></a></td>']
        lines += ['<td rowspan=3 align=center><a href="../stack/plots/spStack-'+starid+'_'+redtag+'_flux.png"><img src="../stack/plots/spStack-'+starid+'_'+redtag+'_flux.png" height=200></a></td>']
        lines += ['</tr>']
        # PSF plots
        lines += ['<tr>']
        lines += ['<td align=center>PSF</td>']        
        for e in range(nexp):
            exp = expnames[e]  
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs1_opsf.png"><img src="../plots/'+starid+'_'+exp+'_nrs1_opsf.png" height=200></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs2_opsf.png"><img src="../plots/'+starid+'_'+exp+'_nrs2_opsf.png" height=200></a></td>']
        lines += ['</tr>']            
        # Flux plots
        lines += ['<tr>']
        lines += ['<td align=center>Flux</td>']        
        for e in range(nexp):            
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs1_flux.png"><img src="../plots/'+starid+'_'+exp+'_nrs1_flux.png" height=200></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs2_flux.png"><img src="../plots/'+starid+'_'+exp+'_nrs2_flux.png" height=200></a></td>']
        lines += ['</tr>']
    
    lines += ['</table>']
    lines += ['</body>']     
    lines += ['</html>']    

    outfile = outdir+obsid+'_qa.html'
    print('Writing QA html file to ',outfile)
    dln.writelines(outfile,lines)

    

def qa_multi(obsid):
    """ Make QA html pages."""

    outdir = obsid+'/qa/'
    if os.path.exists(outdir)==False:
        os.makedirs(outdir)
    
    lines = []
    lines += ['<html>']
    lines += ['<head>']
    lines += ['<title>Observation '+obsid]
    lines += ['</title>']
    lines += ['<base target="_blank">']
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
    lines += ['<h4>'+str(nexp)+' exposures: ']
    lines += ['<ul>']
    for i in range(nexp):
        lines += ['<li>'+expnames[i]]
    lines += ['</ul></h4>']
        
    calstackfiles = glob(obsid+'/stack/spStack-*_cal.fits*')
    starids = [os.path.basename(c)[8:-9] for c in calstackfiles if os.path.getsize(c)>0]
    nstars = len(starids)

    lines += ['<h3>Click on an image to see a larger version</h3><p>']
    
    lines += ['<table border=1>']
    header = '<tr bgcolor=#AEB6BF><th>Number</th><th>Name</th><th>Image Type</th>'
    colors = ['#85C1E9','#F5B041','#BB8FCE','#27AE60','#EC7063','#008080']   # blue,orange,purple,green,red,teal
    for e in range(nexp):
        exp = expnames[e]
        color = colors[e]
        header += '<th colspan=2 bgcolor='+color+'>Exposure '+str(e+1)+' Image</th>'
        header += '<th colspan=2 bgcolor='+color+'>Exposure '+str(e+1)+' PSF</th>'
        header += '<th colspan=2 bgcolor='+color+'>Exposure '+str(e+1)+' Flux</th>'        
    header += '<td align=center>Combined Flux</th>'
    lines += [header]
    lines += ['<tr bgcolor=#AEB6BF><td></td><td></td><td></td>'+3*nexp*'<td align=center>NRS1</td><td align=center>NRS2</td>'+'<td></td></tr>']
    
    # A separate row for each star
    for i in range(nstars):
        starid = starids[i]
        lines += ['<tr>']
        lines += ['<td rowspan=2 align=center>'+str(i+1)+'</td>']
        lines += ['<td rowspan=2 align=center>'+starid+'</td>']
        lines += ['<td align=center>Cal</td>']
        # Cal plots
        for e in range(nexp):
            exp = expnames[e]
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs1_caldata.png"><img src="../plots/'+starid+'_'+exp+'_nrs1_caldata.png" height=200></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs2_caldata.png"><img src="../plots/'+starid+'_'+exp+'_nrs2_caldata.png" height=200></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs1_calopsf.png"><img src="../plots/'+starid+'_'+exp+'_nrs1_calopsf.png" height=200></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs2_calopsf.png"><img src="../plots/'+starid+'_'+exp+'_nrs2_calopsf.png" height=200></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs1_calflux.png"><img src="../plots/'+starid+'_'+exp+'_nrs1_calflux.png" height=200></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs2_calflux.png"><img src="../plots/'+starid+'_'+exp+'_nrs2_calflux.png" height=200></a></td>']
        lines += ['<td><a href="../stack/plots/spStack-'+starid+'_cal_flux.png"><img src="../stack/plots/spStack-'+starid+'_cal_flux.png" height=200></a></td>']
        #lines += ['<td><img src="../stack/plots/spStack-'+starid+'_rate_flux.png"></td>']        
        lines += ['</tr>']
        
        # Rate plots
        lines += ['<tr>']
        lines += ['<td align=center>Rate</td>']
        for e in range(nexp):
            exp = expnames[e]
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs1_ratedata.png"><img src="../plots/'+starid+'_'+exp+'_nrs1_ratedata.png" height=200></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs2_ratedata.png"><img src="../plots/'+starid+'_'+exp+'_nrs2_ratedata.png" height=200></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs1_rateopsf.png"><img src="../plots/'+starid+'_'+exp+'_nrs1_rateopsf.png" height=200></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs2_rateopsf.png"><img src="../plots/'+starid+'_'+exp+'_nrs2_rateopsf.png" height=200></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs1_rateflux.png"><img src="../plots/'+starid+'_'+exp+'_nrs1_rateflux.png" height=200></a></td>']
            lines += ['<td><a href="../plots/'+starid+'_'+exp+'_nrs2_rateflux.png"><img src="../plots/'+starid+'_'+exp+'_nrs2_rateflux.png" height=200></a></td>']            
        lines += ['<td><a href="../stack/plots/spStack-'+starid+'_rate_flux.png"><img src="../stack/plots/spStack-'+starid+'_rate_flux.png" height=200></a></td>']
        #lines += ['<td><img src="../stack/plots/spStack-'+starid+'_rate_flux.png"></td>']        
        lines += ['</tr>']
        
        # a row for Cal and another one for Rate
    
    
    lines += ['</table>']
    lines += ['</body>']     
    lines += ['</html>']    

    outfile = outdir+obsid+'_qa.html'
    print('Writing QA html file to ',outfile)
    dln.writelines(outfile,lines)
