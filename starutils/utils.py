from __future__ import print_function,division

import logging
import numpy as np

from astropy.coordinates import SkyCoord
from .extinction import get_AV_infinity


def get_trilegal(filename,ra,dec,filterset='kepler_2mass',area=1,maglim=27):
    try:
        c = SkyCoord(ra,dec)
    except UnitsError:
        c = SkyCoord(ra,dec,unit='deg')
    l,b = (c.galactic.l.value,c.galactic.b.value)
    outfile = '%s.dat' % (name)
    AV = get_AV_infinity(l,b,frame='galactic')
    cmd = 'get_trilegal 1.6beta %f %f %f %.3f %s 1 %.1f %s' % (l,b,area,AV,filterset,maglim,outfile)
    logging.info('running: {}'.format(cmd))
    sp.Popen(cmd,shell=True).wait()
    shutil.move(outfile,'%s/%s' % (dir,outfile))

