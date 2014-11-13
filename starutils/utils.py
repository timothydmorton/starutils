from __future__ import print_function,division

import os,os.path
import pkg_resources

import numpy as np
import numpy.random as rand

from simpledist.distributions import KDE_Distribution

DATAFOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

RAGHAVAN_PERS = np.recfromtxt('{}/raghavan_periods.dat'.format(DATAFOLDER))
RAGHAVAN_LOGPERS = np.log10(RAGHAVAN_PERS.f1[RAGHAVAN_PERS.f0 == 'Y'])
RAGHAVAN_BINPERS = RAGHAVAN_PERS.f1[RAGHAVAN_PERS.f0 == 'Y']
RAGHAVAN_BINPERKDE = KDE_Distribution(RAGHAVAN_BINPERS,adaptive=False)
RAGHAVAN_LOGPERKDE = KDE_Distribution(RAGHAVAN_LOGPERS,adaptive=False)

#from Multiple Star Catalog
MSC_TRIPDATA = np.recfromtxt('{}/multiple_pecc.txt'.format(DATAFOLDER),names=True)
MSC_TRIPLEPERS = MSC_TRIPDATA.P
MSC_TRIPPERKDE = KDE_Distribution(MSC_TRIPLEPERS,adaptive=False)
MSC_TRIPLOGPERKDE = KDE_Distribution(np.log10(MSC_TRIPLEPERS),adaptive=False)

def randpos_in_circle(n,rad,return_rad=False):
    x = rand.random(n)*2*rad - rad
    y = rand.random(n)*2*rad - rad
    mask = (x**2 + y**2 > rad**2)
    nw = mask.sum()
    while nw > 0:
        x[mask] = rand.random(nw)*2*rad-rad
        y[mask] = rand.random(nw)*2*rad-rad
        mask = (x**2 + y**2 > rad**2)
        nw = mask.sum()
    if return_rad:
        return np.sqrt(x**2 + y**2)
    else:
        return x,y

def draw_pers_eccs(n,**kwargs):
    pers = draw_raghavan_periods(n)
    eccs = draw_eccs(n,pers,**kwargs)
    return pers,eccs

def draw_raghavan_periods(n):
    logps = RAGHAVAN_LOGPERKDE.resample(n)
    return 10**logps

def draw_eccs(n,per=10,binsize=0.1,fuzz=0.05,maxecc=0.97):
    """draws eccentricities appropriate to given periods, generated according to empirical data from Multiple Star Catalog
    """
    if np.size(per) == 1 or np.std(np.atleast_1d(per))==0:
        if np.size(per)>1:
            per = per[0]
        ne=0
        while ne<10:
            #if per > 25:
            #    w = where((TRIPLEPERS>25) & (TRIPLEPERS<300))
            #else:
            #    w = where(abs(TRIPLEPERS-per)<binsize/2.)
            mask = np.absolute(np.log10(MSC_TRIPLEPERS)-np.log10(per))<binsize/2.
            es = MSC_TRIPDATA.e[mask]
            ne = len(es)
            if ne<10:
                binsize*=1.1
        inds = rand.randint(ne,size=n)
        es = es[inds] * (1 + rand.normal(size=n)*fuzz)
    
    else:
        longmask = (per > 25)
        shortmask = (per <= 25)
        es = np.zeros(np.size(per))

        elongs = MSC_TRIPDATA.e[MSC_TRIPLEPERS > 25]
        eshorts = MSC_TRIPDATA.e[MSC_TRIPLEPERS <= 25]

        n = np.size(per)
        nlong = longmask.sum()
        nshort = shortmask.sum()
        nelongs = np.size(elongs)
        neshorts = np.size(eshorts)
        ilongs = rand.randint(nelongs,size=nlong)
        ishorts = rand.randint(neshorts,size=nshort)
        
        es[longmask] = elongs[ilongs]
        es[shortmask] = eshorts[ishorts]

    es = es * (1 + rand.normal(size=n)*fuzz)
    es[es>maxecc] = maxecc
    return np.absolute(es)
