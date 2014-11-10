from __future__ import print_function,division

import os,re
import logging
import subprocess as sp
from astropy.coordinates import SkyCoord

def get_AV_infinity(ra,dec,frame='icrs',verbose=False):
    coords = SkyCoord(ra,dec,unit='deg',frame=frame).transform_to('icrs')

    rah,ram,ras = coords.ra.hms
    decd,decm,decs = coords.dec.dms
    if decd > 0: 
        decsign = '%2B'
    else:
        decsign = ''
    url = 'http://ned.ipac.caltech.edu/cgi-bin/nph-calc?in_csys=Equatorial&in_equinox=J2000.0&obs_epoch=2010&lon='+'%i' % rah + \
        '%3A'+'%i' % ram + '%3A' + '%05.2f' % ras + '&lat=%s' % decsign + '%i' % decd + '%3A' + '%i' % decm + '%3A' + '%05.2f' % decs + \
        '&pa=0.0&out_csys=Equatorial&out_equinox=J2000.0'

    tmpfile = '/tmp/nedsearch%s%s.html' % (ra,dec)
    cmd = 'wget \'%s\' -O %s -q' % (url,tmpfile)
    sp.Popen(cmd,shell=True).wait()
    AV = None
    for line in open(tmpfile,'r'):
        m = re.search('V \(0.54\)\s+(\S+)',line)
        if m:
            AV = float(m.group(1))
    if AV is None:
        logging.warning('Error accessing NED!')
        for line in open(tmpfile):
            logging.warning(line)
        


    os.remove(tmpfile)
    return AV
    
    
    
