from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import logging
import re, os, os.path
import numpy.random as rand

from astropy import units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord

from orbitutils import OrbitPopulation,OrbitPopulation_FromH5
from orbitutils import TripleOrbitPopulation, TripleOrbitPopulation_FromH5
from plotutils import setfig,plot2dhist

from simpledist import distributions as dists

from .constraints import Constraint,UpperLimit,LowerLimit,JointConstraintOr
from .constraints import ConstraintDict,MeasurementConstraint,RangeConstraint
from .constraints import ContrastCurveConstraint,VelocityContrastCurveConstraint

from .utils import randpos_in_circle, draw_raghavan_periods, draw_msc_periods, draw_eccs
from .utils import flat_massratio_fn, mult_masses
from .utils import distancemodulus, addmags, dfromdm

from .trilegal import get_trilegal

try:
    from isochrones.dartmouth import Dartmouth_Isochrone
    DARTMOUTH = Dartmouth_Isochrone()
except ImportError:
    logging.warning('isochrones package not installed; population simulations will not be fully functional')
    DARTMOUTH = None

class StarPopulation(object):
    def __init__(self,stars,distance=None,
                 max_distance=1000*u.pc,convert_absmags=True,
                 name=''):
        """A population of stars.  Initialized with no constraints.

        stars : ``pandas`` ``DataFrame`` object
            Data table containing properties of stars.
            Magnitude properties end with "_mag".  Default
            is that these magnitudes are absolute, and get 
            converted to apparent magnitudes based on distance,
            which is either provided or randomly assigned.

        distance : ``Quantity`` or float, optional
            If not ``None``, then distances of stars are assigned
            randomly out to max_distance.  If float,
            then assumed to be in parsec.  Or, if stars already 
            has a distance column, this is ignored.

        max_distance : ``Quantity`` or float, optional
            Max distance out to which distances will be simulated,
            according to random placements in volume ($p(d)\simd^2$).  
            Ignored if stars already has a distance column.

        convert_absmags : bool
            If ``True``, then magnitudes in ``stars`` will be converted
            to apparent magnitudes based on distance.  If ``False,``
            then magnitudes will be kept as-is.  Ignored if stars already
            has a distance column.

        """
        self.stars = stars.copy()
        self.name = name

        N = len(self.stars)

        #if stars does not have a 'distance' column already, then
        # we define distances based on the provided arguments,
        # and covert absolute magnitudes into apparent (unless explicitly
        # forbidden from doing so by 'convert_absmags' being set
        # to False).
        
        if 'distance' not in self.stars:
            if type(max_distance) != Quantity:
                max_distance = max_distance * u.pc


            if distance is None:
                #generate random distances
                dmax = max_distance.to('pc').value
                distance_distribution = dists.PowerLaw_Distribution(2.,1,dmax) # p(d)~d^2
                distance = distance_distribution.rvs(N)

            if type(distance) != Quantity:
                distance = distance * u.pc

            distmods = distancemodulus(distance)
            if convert_absmags:
                for col in self.stars.columns:
                    if re.search('_mag',col):
                        self.stars[col] += distmods

            self.stars['distance'] = distance
            self.stars['distmod'] = distmods


        if 'distmod' not in self.stars:
            self.stars['distmod'] = distancemodulus(self.stars['distance'])

        #initialize empty constraint list
        self.constraints = ConstraintDict()
        self.hidden_constraints = ConstraintDict()
        self.selectfrac_skip = []
        self.distribution_skip = []

        #apply constraints,  initializing the following attributes:
        # self.distok, self.countok, self.selected, self.selectfrac
        
        self._apply_all_constraints()


    def __getitem__(self,prop):
        return self.selected[prop]

    @property
    def magnitudes(self):
        bands = []
        for c in self.stars.columns:
            if re.search('_mag',c):
                bands.append(c)
        return bands

    @property
    def distance(self):
        return np.array(self.stars['distance'])*u.pc

    @distance.setter
    def distance(self,value):
        """value must be a ``Quantity`` object
        """
        self.stars['distance'] = value.to('pc').value

        old_distmod = self.stars['distmod'].copy()
        new_distmod = distancemodulus(self.stars['distance'])

        for m in self.magnitudes:
            self.stars[m] += new_distmod - old_distmod

        self.stars['distmod'] = new_distmod

        logging.warning('Setting the distance manually may have screwed up your constraints.  Re-apply constraints as necessary.')


    def _apply_all_constraints(self):
        """Applies all constraints in ``self.constraints`` to population.

        A constraint may either change the overall character of the population,
        or just the overall number of allowed stars.  If a constraint
        changes just the overall number of allowed stars (not the makeup),
        then the name of the constraint should be in the ``self.distribution_skip``
        list.  If the opposite is true, then the constraint name should be in 
        ``self.selectfrac_skip``.

        Result
        ------
        Sets values for ``self.distok``, ``self.countok``, 
        ``self.selected``, and ``self.selectfrac`` attributes.
        """

        n = len(self.stars)
        self.distok = np.ones(len(self.stars)).astype(bool)
        self.countok = np.ones(len(self.stars)).astype(bool)

        for name in self.constraints:
            c = self.constraints[name]
            if c.name not in self.distribution_skip:
                self.distok &= c.ok
            if c.name not in self.selectfrac_skip:
                self.countok &= c.ok

        self.selected = self.stars[self.distok]
        self.selectfrac = self.countok.sum()/n


    def prophist2d(self,propx,propy,
                   logx=False,logy=False,inds=None,
                   fig=None,selected=False,**kwargs):
        """Makes a 2d density histogram of two given properties

        propx, propy : string
            Names of properties to histogram.  Must be names of columns
            in ``self.stars`` table.

        inds : ndarray, optional
            Desired indices of ``self.stars`` to plot.  If ``None``,
            then all is assumed.

        fig : None or int, optional
            Argument passed to ``plotutils.setfig`` function.

        selected : bool, optional
            If ``True``, then only the "selected" stars (that is, stars
            obeying all constraints attached to this object) will
            be plotted.

        kwargs :
            Keyword arguments passed to ``plot2dhist`` function.
        """
        if inds is None:
            if selected:
                inds = np.arange(len(self.selected))
            else:
                inds = np.arange(len(self.stars))
        if selected:
            xvals = self[propx].iloc[inds]
            yvals = self[propy].iloc[inds]
        else:
            xvals = self.stars[propx].iloc[inds]
            yvals = self.stars[propy].iloc[inds]
        if logx:
            xvals = np.log10(xvals)
        if logy:
            yvals = np.log10(yvals)

        plot2dhist(xvals,yvals,fig=fig,**kwargs)
        plt.xlabel(propx)
        plt.ylabel(propy)
        

    def prophist(self,prop,fig=None,log=False,inds=None,
                 selected=False,**kwargs):
        """Plots a histogram of desired property
        """
        
        setfig(fig)
        if inds is None:
            if selected:
                inds = np.arange(len(self.selected))
            else:
                inds = np.arange(len(self.stars))
        if selected:
            vals = self[prop].iloc[inds]
        else:
            vals = self.stars[prop].iloc[inds]

        if log:
            h = plt.hist(np.log10(vals),**kwargs)
        else:
            h = plt.hist(vals,**kwargs)
        
        plt.xlabel(prop)

    def constraint_stats(self,primarylist=None):
        """Returns information about effect of constraints on population.
        """
        if primarylist is None:
            primarylist = []
        n = len(self.stars)
        primaryOK = np.ones(n).astype(bool)
        tot_reject = np.zeros(n)

        for name in self.constraints:
            if name in self.selectfrac_skip:
                continue
            c = self.constraints[name]
            if name in primarylist:
                primaryOK &= c.ok
            tot_reject += ~c.ok
        primary_rejected = ~primaryOK
        secondary_rejected = tot_reject - primary_rejected
        lone_reject = {}
        for name in self.constraints:
            if name in primarylist or name in self.selectfrac_skip:
                continue
            c = self.constraints[name]
            lone_reject[name] = ((secondary_rejected==1) & (~primary_rejected) & (~c.ok)).sum()/float(n)
        mult_rejected = (secondary_rejected > 1) & (~primary_rejected)
        not_rejected = ~(tot_reject.astype(bool))
        primary_reject_pct = primary_rejected.sum()/float(n)
        mult_reject_pct = mult_rejected.sum()/float(n)
        not_reject_pct = not_rejected.sum()/float(n)
        tot = 0

        results = {}
        results['pri'] = primary_reject_pct
        tot += primary_reject_pct
        for name in lone_reject:
            results[name] = lone_reject[name]
            tot += lone_reject[name]
        results['multiple constraints'] = mult_reject_pct
        tot += mult_reject_pct
        results['remaining'] = not_reject_pct
        tot += not_reject_pct

        if tot != 1:
            logging.warning('total adds up to: %.2f (%s)' % (tot,self.model))

        return results

    
    def constraint_piechart(self,primarylist=None,
                            fig=None,title='',colordict=None,
                            legend=True,nolabels=False):
        """Makes piechart illustrating constraints on population

        for FPP, primarylist default was ['secondary depth']; remember that
        """

        setfig(fig,figsize=(6,6))
        stats = self.constraint_stats(primarylist=primarylist)
        if primarylist is None:
            primarylist = []
        if len(primarylist)==1:
            primaryname = primarylist[0]
        else:
            primaryname = ''
            for name in primarylist:
                primaryname += '%s,' % name
            primaryname = primaryname[:-1]
        fracs = []
        labels = []
        explode = []
        colors = []
        fracs.append(stats['remaining']*100)
        labels.append('remaining')
        explode.append(0.05)
        colors.append('b')
        if 'pri' in stats and stats['pri']>=0.005:
            fracs.append(stats['pri']*100)
            labels.append(primaryname)
            explode.append(0)
            if colordict is not None:
                colors.append(colordict[primaryname])
        for name in stats:
            if name == 'pri' or \
                    name == 'multiple constraints' or \
                    name == 'remaining':
                continue

            fracs.append(stats[name]*100)
            labels.append(name)
            explode.append(0)
            if colordict is not None:
                colors.append(colordict[name])
            
        if stats['multiple constraints'] >= 0.005:
            fracs.append(stats['multiple constraints']*100)
            labels.append('multiple constraints')
            explode.append(0)
            colors.append('w')

        autopct = '%1.1f%%'

        if nolabels:
            labels = None
        if legend:
            legendlabels = []
            for i,l in enumerate(labels):
                legendlabels.append('%s (%.1f%%)' % (l,fracs[i]))
            labels = None
            autopct = ''
        if colordict is None:
            plt.pie(fracs,labels=labels,autopct=autopct,explode=explode)
        else:
            plt.pie(fracs,labels=labels,autopct=autopct,explode=explode,
                    colors=colors)
        if legend:
            plt.legend(legendlabels,bbox_to_anchor=(-0.05,0),
                       loc='lower left',prop={'size':10})
        plt.title(title)

    def apply_constraint(self,constraint,selectfrac_skip=False,
                         distribution_skip=False,overwrite=False):
        """Apply a constraint to the population
        """
        if constraint.name in self.constraints and not overwrite:
            logging.warning('constraint already applied: {}'.format(constraint.name))
            return
        self.constraints[constraint.name] = constraint
        if selectfrac_skip:
            self.selectfrac_skip.append(constraint.name)
        if distribution_skip:
            self.distribution_skip.append(constraint.name)

        self._apply_all_constraints()

    def replace_constraint(self,name,selectfrac_skip=False,distribution_skip=False):
        """Re-apply constraint that had been removed
        """
        if name in self.hidden_constraints:
            c = self.hidden_constraints[name]
            self.apply_constraint(c,selectfrac_skip=selectfrac_skip,
                                  distribution_skip=distribution_skip)
            del self.hidden_constraints[name]
        else:
            logging.warning('Constraint {} not available for replacement.'.format(name))

    def remove_constraint(self,name):
        """Remove a constraint (make it "hidden")
        """
        if name in self.constraints:
            self.hidden_constraints[name] = self.constraints[name]
            del self.constraints[name]
            if name in self.distribution_skip:
                self.distribution_skip.remove(name)
            if name in self.selectfrac_skip:
                self.selectfrac_skip.remove(name)
            self._apply_all_constraints()
        else:
            logging.warning('Constraint {} does not exist.'.format(name))

    def constrain_property(self,prop,lo=-np.inf,hi=np.inf,
                           measurement=None,thresh=3,
                           selectfrac_skip=False,distribution_skip=False):
        """Apply constraint that constrains property.

        prop : string
            Name of property.  Must be column in ``self.stars``.

        lo,hi : float, optional
            Low and high allowed values for ``prop``.

        measurement : (value,err)
            Value and error of measurement. 

        thresh : float
            Number of "sigma" to allow.

        selectfrac_skip, distribution_skip : bool
            Passed to ``self.apply_constraint``
        """
        if prop in self.constraints:
            logging.info('re-doing {} constraint'.format(prop))
            self.remove_constraint(prop)
        if measurement is not None:
            val,dval = measurement
            self.apply_constraint(MeasurementConstraint(getattr(self.stars,prop),
                                                        val,dval,name=prop,thresh=thresh),
                                  selectfrac_skip=selectfrac_skip,
                                  distribution_skip=distribution_skip)
        else:
            self.apply_constraint(RangeConstraint(getattr(self.stars,prop),
                                                  lo=lo,hi=hi,name=prop),
                                  selectfrac_skip=selectfrac_skip,
                                  distribution_skip=distribution_skip)

    def apply_trend_constraint(self,limit,dt,**kwargs):
        """Only works if object has dRV method and plong attribute; limit in km/s

        limit : ``Quantity``
            Radial velocity limit on trend

        dt : ``Quantity``
            Time baseline of RV observations.
        """
        dRVs = np.absolute(self.dRV(dt))
        c1 = UpperLimit(dRVs, limit)
        c2 = LowerLimit(self.Plong, dt*4)

        self.apply_constraint(JointConstraintOr(c1,c2,name='RV monitoring',
                                                Ps=self.Plong,dRVs=dRVs),**kwargs)

    def apply_cc(self,cc,**kwargs):
        """Only works if object has Rsky, dmag attributes
        """
        rs = self.Rsky.to('arcsec').value
        dmags = self.dmag(cc.band)
        self.apply_constraint(ContrastCurveConstraint(rs,dmags,cc,name=cc.name),
                              **kwargs)

    def apply_vcc(self,vcc,**kwargs):
        """only works if has dmag and RV attributes"""
        rvs = self.RV.value
        dmags = self.dmag(vcc.band)
        self.apply_constraint(VelocityContrastCurveConstraint(rvs,dmags,vcc,
                                                              name='secondary spectrum'),
                              **kwargs)
        
    def set_maxrad(self,maxrad):
        """Adds a constraint that rejects everything with Rsky > maxrad

        Requires Rsky attribute, which should always have units.

        Parameters
        ----------
        maxrad : ``Quantity``
            The maximum angular value of Rsky.
        """
        self.maxrad = maxrad
        self.apply_constraint(UpperLimit(self.Rsky,maxrad,
                                         name='Max Rsky'),
                              overwrite=True)
        self._apply_all_constraints()
        

    @property
    def constraint_df(self):
        """A ``DataFrame`` representing all constraints, hidden or not
        """
        df = pd.DataFrame()
        for name,c in self.constraints.iteritems():
            df[name] = c.ok
        for name,c in self.hidden_constraints.iteritems():
            df[name] = c.ok
        return df

    def save_hdf(self,filename,path='',properties=None):
        if properties is None:
            properties = {}
        self.stars.to_hdf(filename,'{}/stars'.format(path))
        self.constraint_df.to_hdf(filename,'{}/constraints'.format(path))

        store = pd.HDFStore(filename)
        attrs = store.get_storer('{}/stars'.format(path)).attrs
        attrs.selectfrac_skip = self.selectfrac_skip
        attrs.distribution_skip = self.distribution_skip
        attrs.name = self.name
        attrs.poptype = type(self)
        attrs.properties = properties
        store.close()

class StarPopulation_FromH5(StarPopulation):
    def __init__(self,filename,path=''):
        """Loads in a StarPopulation saved to .h5
        """
        stars = pd.read_hdf(filename,path+'/stars')
        constraint_df = pd.read_hdf(filename,path+'/constraints')
        store = pd.HDFStore(filename)
        attrs = store.get_storer('{}/stars'.format(path)).attrs
        distribution_skip = attrs.distribution_skip
        selectfrac_skip = attrs.selectfrac_skip
        name = attrs.name
        poptype = attrs.poptype
        for kw,val in attrs.properties.items():
            setattr(self,kw,val)
        store.close()

        self.poptype = poptype
        StarPopulation.__init__(self,stars,name)

        for n in constraint_df.columns:
            mask = np.array(constraint_df[n])
            c = Constraint(mask,name=n)
            sel_skip = n in selectfrac_skip
            dist_skip = n in distribution_skip
            self.apply_constraint(c,selectfrac_skip=sel_skip,
                                  distribution_skip=dist_skip)

class BinaryPopulation(StarPopulation):
    def __init__(self,primary,secondary,
                 orbpop=None, period=None,
                 ecc=None,
                 is_single=None,
                 **kwargs):

        """A population of binary stars.

        If ``OrbitPopulation`` provided, that will describe the orbits;
        if not, then orbit population will be generated.  Single stars may
        be indicated if desired by having their mass set to zero and all
        magnitudes set to ``inf``.

        Parameters
        ----------
        primary,secondary : ``DataFrame``
            Properties of primary and secondary stars, respectively.
            These get merged into new ``stars`` attribute, with "_A"
            and "_B" tags.   

        orbpop : ``OrbitPopulation``, optional
            Object describing orbits of stars.  If not provided, then ``period``
            and ``ecc`` keywords must be provided, or else they will be
            randomly generated (see below).

        period,ecc : array-like, optional
            Periods and eccentricities of orbits.  If ``orbpop``
            not passed, and these are not provided, then periods and eccs 
            will be randomly generated according
            to the empirical distributions of the Raghavan (2010) and
            Multiple Star Catalog distributions (see ``utils`` for details).

        """

        assert len(primary)==len(secondary)

        stars = pd.DataFrame()

        for c in primary.columns:
            if re.search('_mag',c):
                stars[c] = addmags(primary[c],secondary[c])
            stars['{}_A'.format(c)] = primary[c]
        for c in secondary.columns:
            stars['{}_B'.format(c)] = secondary[c]            
            
        stars['q'] = stars['mass_B']/stars['mass_A']


        if orbpop is None:
            if period is None:
                period = draw_raghavan_periods(len(secondary))
            if ecc is None:
                ecc = draw_eccs(len(secondary),period)
            self.orbpop = OrbitPopulation(primary['mass'],
                                          secondary['mass'],
                                          period,ecc)
        else:
            self.orbpop = orbpop

        StarPopulation.__init__(self,stars,**kwargs)


    @property
    def singles(self):
        return self.stars.query('mass_B == 0')

    @property
    def binaries(self):
        return self.stars.query('mass_B > 0')

    def binary_fraction(self,query='mass_A >= 0'):
        subdf = self.stars.query(query)
        nbinaries = (subdf['mass_B'] > 0).sum()
        frac = nbinaries/len(subdf)
        return frac, frac/np.sqrt(nbinaries)
        
    @property
    def Rsky(self):
        r = (self.orbpop.Rsky/self.distance)
        return r.to('arcsec',equivalencies=u.dimensionless_angles())

    @property
    def RV(self):
        return self.orbpop.RV

    def dRV(self,dt):
        return self.orbpop.dRV(dt)

    @property
    def Plong(self):
        return self.orbpop.P

    def dmag(self,band):
        mag2 = self.stars['{}_mag_B'.format(band)]
        mag1 = self.stars['{}_mag_A'.format(band)]
        return mag2-mag1

    def rsky_distribution(self,rmax=None,dr=0.005,smooth=0.1,nbins=100):
        if rmax is None:
            if hasattr(self,'maxrad'):
                rmax = self.maxrad
            else:
                rmax = np.percentile(self.Rsky,99)
        rs = np.arange(0,rmax,dr)
        dist = dists.Hist_Distribution(self.Rsky.value,bins=nbins,maxval=rmax,smooth=smooth)
        return dist

    def rsky_lhood(self,rsky,**kwargs):
        dist = self.rsky_distribution(**kwargs)
        return dist(rsky)

    def save_hdf(self,filename,path=''):
        self.orbpop.save_hdf(filename,path='{}/orbpop'.format(path))
        StarPopulation.save_hdf(self,filename,path=path)

class VolumeLimitedPopulation(BinaryPopulation):
    def __init__(self, m1, dmax, n=1e5, binary_fraction=0.4,
                 minmass=0.11, ichrone=DARTMOUTH,
                 age=9.7, feh=0.0, **kwargs): 
        """A volume limited sample of stars ***OBSELETE***

        Parameters
        ----------
        
        m1 : array_like
            Primary masses

        dmax : ``Quantity`` or float
            Maximum distance of sample.

        n : integer, optional
            Size of population

        binary_fraction : float
            Fraction of stars that should be binary.

        """

        qmin = minmass/m1
        q = rand.random(n)*(1-qmin) + qmin
        m2 = m1*q
        
        primaries = ichrone(m1,age,feh)
        secondaries = ichrone(m2,age,feh)
        
        ubin = rand.random(n)
        is_single = ubin > binary_fraction
        for c in secondaries.columns:
            if re.search('_mag',c):
                secondaries[c][is_single] = np.inf
            else:
                secondaries[c][is_single] = np.nan

        #generate random distances
        distance_distribution = dists.PowerLaw_Distribution(2.,1,dmax) # p(d)~d^2
        distances = distance_distribution.rvs(n)

        BinaryPopulation.__init__(self,primaries,secondaries,distances,**kwargs)

        self.stars['is_binary'] = ~is_single

    @property
    def singles(self):
        return self.stars.query('not is_binary')

    @property
    def binaries(self):
        return self.stars.query('is_binary')


    def binary_fraction(self,query='is_binary or not is_binary', unc=False):
        subdf = self.stars.query(query)
        frac = subdf['is_binary'].sum()/len(subdf)
        if unc:
            return frac, frac/np.sqrt(subdf['is_binary'].sum())
        else:
            return frac
        
class Simulated_BinaryPopulation(BinaryPopulation):
    def __init__(self,M,q_fn,P_fn,ecc_fn,n=1e4,ichrone=DARTMOUTH,
                 age=9.5,feh=0.0, minmass=0.12, **kwargs):
        """Simulates BinaryPopulation according to provide primary mass(es), generating functions, and stellar isochrone models.

        Parameters
        ----------
        M : float or array-like
            Primary mass(es).

        q_fn : function
            Mass ratio generating function. Must return 'n' mass ratios, and be
            called as follows::
        
                qs = q_fn(n)

        P_fn : function
            Orbital period generating function.  Must return ``n`` orbital periods,
            and be called as follows::
            
                Ps = P_fn(n)

        ecc_fn : function
            Orbital eccentricity generating function.  Must return ``n`` orbital 
            eccentricities generated according to provided period(s)::

                eccs = ecc_fn(n,Ps)

        n : int, optional
            Number of instances to simulate.

        ichrone : ``Isochrone`` object
            Stellar model object from which to simulate stellar properties.

        age,feh : float or array-like
            log(age) and metallicity at which to simulate population.
        """
        M2 = M * q_fn(n)
        P = P_fn(n)
        ecc = ecc_fn(n,P)


        pri = ichrone(np.ones(n)*M, age, feh, return_df=True) #array typecast needed b/c of isochrones bug; fix at some point.
        sec = ichrone(M2, age, feh, return_df=True)

        BinaryPopulation.__init__(self, pri, sec,
                                  period=P, ecc=ecc, **kwargs)

class Raghavan_BinaryPopulation(Simulated_BinaryPopulation):
    def __init__(self,M,e_M=0,n=1e4,ichrone=DARTMOUTH,
                 age=9.5, feh=0.0, q_fn=None, qmin=0.1,
                 minmass=0.12, **kwargs):
        """A Simulated_BinaryPopulation with empirical default distributions.

        Default mass ratio distribution is flat down to chosen minimum mass,
        default period distribution is from Raghavan (2010), default
        eccentricity/period relation comes from data from the Multiple Star
        Catalog (Tokovinin, xxxx).

        Parameters
        ----------
        M : float or array-like
            Primary mass(es) in solar masses.

        e_M : float, optional
            1-sigma uncertainty in primary mass.

        n : int
            Number of simulated instances to create.

        ichrone : ``Isochrone`` object
            Stellar models from which to generate binary companions.

        age,feh : float or array-like
            Age and metallicity of system.

        name : str
            Name of population.

        q_fn : function
            A function that returns random mass ratios.  Defaults to flat
            down to provided minimum mass.  Must be called as follows::
            
                qs = q_fn(n)

            to provide ``n`` random mass ratios.


        """
        if q_fn is None:
            q_fn = flat_massratio_fn(qmin=max(qmin,minmass/M))

        if e_M != 0:
            M = stats.norm(M,e_M).rvs(n)

        Simulated_BinaryPopulation.__init__(self,M, q_fn,
                                            draw_raghavan_periods,
                                            draw_eccs, n=n,
                                            ichrone=ichrone,
                                            age=age, feh=feh,
                                            minmass=minmass, **kwargs)


class BinaryPopulation_FromH5(BinaryPopulation,StarPopulation_FromH5):
    def __init__(self,filename,path=''):
        """Loads in a BinaryPopulation saved to .h5
        """
        StarPopulation_FromH5.__init__(self,filename,path=path)
        self.orbpop = OrbitPopulation_FromH5(filename,path='{}/orbpop'.format(path))

class VolumeLimitedPopulation_FromH5(VolumeLimitedPopulation,BinaryPopulation_FromH5):
    def __init__(self,filename,path=''):
        """Loads in a VolumeLimitedPopulation saved to .h5 ***OBSELETE?***
        """
        BinaryPopulation_FromH5.__init__(self,filename,path=path)



class TriplePopulation(StarPopulation):
    def __init__(self, primary, secondary, tertiary, 
                 orbpop=None, 
                 period_short=None, period_long=None,
                 ecc_short=0, ecc_long=0,
                 **kwargs):
        """A population of triple stars.

        Primary orbits (secondary + tertiary) in a long orbit;
        secondary and tertiary orbit each other with a shorter orbit.
        Single or double stars may be indicated if desired by having
        the masses of secondary or tertiary set to zero, and all magnitudes
        to ``inf``.
        
        Parameters
        ----------
        primary, secondary, tertiary : ``pandas.DataFrame`` objects
            Properties of primary, secondary, and tertiary stars.
            These will get merged into a new ``stars`` attribute,
            with "_A", "_B", and "_C" tags.

        orbpop : ``OrbitPopulation``, optional
            Object describing orbits of stars.  If not provided, then the period
            and eccentricity keywords must be provided, or else they will be
            randomly generated (see below).

        period_short, period_long, ecc_short, ecc_long : array-like, optional
            Orbital periods and eccentricities of short and long-period orbits. 
            "Short" describes the close pair of the hierarchical system; "long"
            describes the separation between the two major components.  Randomly
            generated if not provided.

            
        """
        assert len(primary)==len(secondary) and len(primary)==len(tertiary)
        N = len(primary)

        stars = pd.DataFrame()

        for c in primary.columns:
            if re.search('_mag',c):
                 stars[c] = addmags(primary[c],secondary[c],tertiary[c])
            stars['{}_A'.format(c)] = primary[c]
        for c in secondary.columns:
            stars['{}_B'.format(c)] = secondary[c]
        for c in tertiary.columns:
            stars['{}_C'.format(c)] = tertiary[c]
               

        ##For orbit population, stars 2 and 3 are in short orbit, and star 1 in long.
        ## So we need to define the proper mapping from A,B,C to 1,2,3.
        ## If C is with A, then A=2, C=3, B=1
        ## If C is with B, then A=1, B=2, C=3

        #CwA = stars['C_orbits']=='A'
        #CwB = stars['C_orbits']=='B'
        #stars['orbpop_number_A'] = np.ones(N)*(CwA*2 + CwB*1)
        #stars['orbpop_number_B'] = np.ones(N)*(CwA*1 + CwB*2)
        #stars['orbpop_number_C'] = np.ones(N)*3

        if orbpop is None:
            if period_long is None or period_short is None:
                period_1 = draw_raghavan_periods(N)
                period_2 = draw_msc_periods(N)                
                period_short = np.minimum(period_1, period_2)
                period_long = np.maximum(period_1, period_2)

            if ecc_short is None or ecc_long is None:
                ecc_short = draw_eccs(N,period_short)
                ecc_long = draw_eccs(N,period_long),
            
            M1 = stars['mass_A']
            M2 = stars['mass_B']
            M3 = stars['mass_C']

            self.orbpop = TripleOrbitPopulation(M1,M2,M3,period_long,period_short,
                                                ecclong=ecc_long, eccshort=ecc_short)
        else:
            self.orbpop = orbpop

        StarPopulation.__init__(self,stars,**kwargs)

    @property
    def singles(self):
        return self.stars.query('mass_B==0 and mass_C==0')

    @property
    def binaries(self):
        return self.stars.query('mass_B > 0 and mass_C==0')

    @property
    def triples(self):
        return self.stars.query('mass_B > 0 and mass_C > 0')
        
    def binary_fraction(self,query='mass_A > 0', unc=False):
        subdf = self.stars.query(query)
        nbinaries = ((subdf['mass_B'] > 0) & (subdf['mass_C']==0)).sum()
        frac = nbinaries/len(subdf)
        if unc:
            return frac, frac/np.sqrt(nbinaries)
        else:
            return frac
        
    def triple_fraction(self,query='mass_A > 0', unc=False):
        subdf = self.stars.query(query)
        ntriples = ((subdf['mass_B'] > 0) & (subdf['mass_C'] > 0)).sum()
        frac = ntriples/len(subdf)
        if unc:
            return frac, frac/np.sqrt(ntriples)
        else:
            return frac
        
class MultipleStarPopulation(TriplePopulation):
    def __init__(self, m1, age=9.6, feh=0.0,
                 f_binary=0.4, f_triple=0.12,
                 minq=0.1, minmass=0.11,
                 n=1e5, ichrone=DARTMOUTH,
                 multmass_fn=mult_masses,
                 period_long_fn=draw_raghavan_periods,
                 period_short_fn=draw_msc_periods,
                 ecc_fn=draw_eccs,orbpop=None,
                 **kwargs):
        """A population of single, double, and triple stars, generated according to prescription.

        
        """
        m1, m2, m3 = multmass_fn(m1, f_binary=f_binary,
                                 f_triple=f_triple,
                                 minq=minq, minmass=minmass,
                                 n=n)

        #generate stellar properties
        primary = ichrone(m1,age,feh)
        secondary = ichrone(m2,age,feh)
        tertiary = ichrone(m3,age,feh)
        
        #clean up columns that become nan when called with mass=0
        # Remember, we want mass=0 and mags=inf when something doesn't exist
        no_secondary = (m2==0)
        no_tertiary = (m3==0)
        for c in secondary.columns: #
            if re.search('_mag',c):
                secondary[c][no_secondary] = np.inf
                tertiary[c][no_tertiary] = np.inf
        secondary['mass'][no_secondary] = 0
        tertiary['mass'][no_tertiary] = 0

        period_1 = period_long_fn(n)
        period_2 = period_short_fn(n)
        period_short = np.minimum(period_1, period_2)
        period_long = np.maximum(period_1, period_2)

        ecc_short = draw_eccs(n, period_short)
        ecc_long = draw_eccs(n, period_long)

        TriplePopulation.__init__(self, primary, secondary, tertiary,
                                  orbpop=orbpop,
                                  period_short=period_short,
                                  period_long=period_long,
                                  ecc_short=ecc_short,
                                  ecc_long=ecc_long,**kwargs)
                        
class BGStarPopulation(StarPopulation):
    def __init__(self,stars,mags=None,maxrad=1800,density=None,name=''):
        """Background star population

        Parameters
        ----------
        stars : ``pandas.DataFrame``
            Properties of stars.  Must have 'distance' column defined.

        """
        if 'distance' not in stars:
            raise ValueError('Stars must have distance column defined')

        self.mags = mags

        if density is None:
            self.density = len(stars)/((3600.*u.arcsec)**2) #default is for TRILEGAL sims to be 1deg^2
        else:
            if type(density)!=Quantity:
                raise ValueError('Provided stellar density must have units.')
            self.density = density
        
        if type(maxrad) != Quantity:
            self._maxrad = maxrad*u.arcsec #arcsec
        else:
            self._maxrad = maxrad

        StarPopulation.__init__(self,stars,name=name)
        self.stars['Rsky'] = randpos_in_circle(len(stars),maxrad,return_rad=True)
        
    @property
    def Rsky(self):
        return np.array(self.stars['Rsky'])*u.arcsec

    @property
    def maxrad(self):
        return self._maxrad

    @maxrad.setter
    def maxrad(self,value):
        if type(value) != Quantity:
            value = value*u.arcsec
        self.stars['Rsky'] *= (value/self._maxrad).decompose()
        self._maxrad = value
        
    def dmag(self,band):
        if self.mags is None:
            raise ValueError('dmag is not defined because primary mags are not defined for this population.')
        return self.stars['{}_mag'.format(band)] - self.mags[band]
        
    def save_hdf(self,filename,path='',properties=None):
        if properties is None:
            properties = {}
        properties['_maxrad'] = self._maxrad
        properties['density'] = self.density
        StarPopulation.save_hdf(self,filename,path=path,properties=properties)

class BGStarPopulation_FromH5(BGStarPopulation,StarPopulation_FromH5):
    def __init__(self,filename,path=''):
        """Loads in a BGStarPopulation saved to .h5
        """
        StarPopulation_FromH5.__init__(self,filename,path=path)
        #store = pd.HDFStore(filename)
        #properties = store.get_storer('{}/stars'.format(path)).attrs.properties
        #self._maxrad = properties['_maxrad']
        #self.density = properties['density']
        #store.close()


#BUGGY RIGHT NOW, FIX!
class BGStarPopulation_TRILEGAL(BGStarPopulation):
    def __init__(self,filename,ra,dec,mags=None,maxrad=1800,
                 name='',**kwargs):
        """Creates TRILEGAL simulation for ra,dec; loads as BGStarPopulation

        keyword arguments passed to ``get_trilegal``
        """
        self.ra = ra
        self.dec = dec
        try:
            stars = pd.read_hdf(filename,'df')
        except:
            get_trilegal(filename,ra,dec,**kwargs)
            stars = pd.read_hdf('{}.h5'.format(filename),'df')
        store = pd.HDFStore(filename)
        self.trilegal_args = store.get_storer('df').attrs.trilegal_args
        store.close()
        area = self.trilegal_args['area']*(u.deg)**2
        density = len(stars)/area

        stars['distmod'] = stars['m-M0']
        stars['distance'] = dfromdm(stars['distmod']) 

        BGStarPopulation.__init__(self,stars,mags=mags,maxrad=maxrad,
                                  density=density,name=name)

    def save_hdf(self,filename,path=''):
        properties = {'trilegal_args':self.trilegal_args,
                      'ra':self.ra,'dec':self.dec}
        BGStarPopulation.save_hdf(self,filename,path=path,
                                  properties=properties)



#methods below should be applied to relevant subclasses
'''
    def set_dmaglim(self,dmaglim):
        if not (hasattr(self,'blendmag') and hasattr(self,'dmaglim')):
            return
        self.dmaglim = dmaglim
        self.apply_constraint(LowerLimit(self.dmags(),self.dmaglim,name='bright blend limit'),overwrite=True)
        self._apply_all_constraints()  #not necessary?
'''

