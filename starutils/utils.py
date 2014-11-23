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

def flat_massratio_fn(qmin=0.1,qmax=1.):
    def fn(n):
        return rand.uniform(size=n)*(qmax - qmin) + qmin
    return fn

def draw_raghavan_periods(n):
    logps = RAGHAVAN_LOGPERKDE.resample(n)
    return 10**logps

def draw_msc_periods(n):
    logps = MSC_TRIPLOGPERKDE.resample(n)
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

########## other utility functions; copied from old code

import astropy.constants as const
AU = const.au.cgs.value
RSUN = const.R_sun.cgs.value
MSUN = const.M_sun.cgs.value
DAY = 86400 #seconds
G = const.G.cgs.value

def rochelobe(q):
    """returns r1/a; q = M1/M2"""
    return 0.49*q**(2./3)/(0.6*q**(2./3) + np.log(1+q**(1./3)))

def withinroche(semimajors,M1,R1,M2,R2):
    q = M1/M2
    return ((R1+R2)*RSUN) > (rochelobe(q)*semimajors*AU)
    
def semimajor(P,mstar=1):
    """Period in days, mstar in solar masses
    """
    return ((P*DAY/2/pi)**2*G*mstar*MSUN)**(1./3)/AU

def period_from_a(a,mstar):
    return np.sqrt(4*pi**2*(a*AU)**3/(G*mstar*MSUN))/DAY

def addmags(*mags):
    tot=0
    for mag in mags:
        tot += 10**(-0.4*mag)
    return -2.5*log10(tot)

def dfromdm(dm):
    if size(dm)>1:
        dm = np.atleast_1d(dm)
    return 10**(1+dm/5)

def distancemodulus(d):
    """d in parsec
    """
    if np.size(d)>1:
        d = np.atleast_1d(d)
    return 5*np.log10(d/10)

def fbofm(M):
    return 0.45 - (0.7-M)/4
