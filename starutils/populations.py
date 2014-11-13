from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging

from astropy import units as u
from astropy.units import Quantity

from orbitutils import OrbitPopulation,OrbitPopulation_FromH5
from plotutils import setfig,plot2dhist

from .constraints import Constraint,UpperLimit,LowerLimit,JointConstraintOr
from .constraints import ConstraintDict,MeasurementConstraint,RangeConstraint
from .constraints import ContrastCurveConstraint,VelocityContrastCurveConstraint

from .utils import randpos_in_circle

class StarPopulation(object):
    def __init__(self,stars,name=''):
        """A population of stars.  Initialized with no constraints.

        data : ``pandas`` ``DataFrame`` object
            Data table containing properties of stars.
            Magnitude properties end with "_mag".

        orbpop : ``OrbitPopulation`` object

        """
        self.stars = stars
        self.name = name
        

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

        Requires Rsky attribute.

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
    def __init__(self,primary,secondary,distance,
                 orbpop=None, period=None,
                 ecc=None,name='',**kwargs):

        """A population of binary stars.

        If ``OrbitPopulation`` provided, that will describe the orbits;
        if not, then orbit population will be generated.

        primary,secondary : ``DataFrame``
            Properties of primary and secondary stars, respectively.
            These get merged into new ``stars`` attribute, with "_A"
            and "_B" tags.

        distance : ``Quantity``
            Distance of system.

        orbpop : ``OrbitPopulation``, optional
            Object describing orbits of stars.  If not provided, then ``period``
            and ``ecc`` keywords must be provided.

        period,ecc : array-like, optional
            Periods and eccentricities of orbits.  Must be provided if ``orbpop``
            not passed.
        """
        stars = pd.DataFrame()
        for c in primary.columns:
            stars['{}_A'.format(c)] = primary[c]
        for c in secondary.columns:
            stars['{}_B'.format(c)] = secondary[c]
        
        stars['distance'] = distance.to('pc').value
        #self.distance = distance

        if orbpop is None:
            self.orbpop = OrbitPopulation(primary['mass'],
                                          secondary['mass'],
                                          period,ecc)
        else:
            self.orbpop = orbpop

        StarPopulation.__init__(self,stars,name=name)

    @property
    def distance(self):
        return np.array(self.stars['distance'])*u.pc
        
    @distance.setter
    def distance(self,value):
        """value must be a ``Quantity`` object
        """
        self.stars['distance'] = value.to('pc').value
        logging.warning('Setting the distance manually may have screwed up your constraints.  Re-apply constraints as necessary.')

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

    def save_hdf(self,filename,path=''):
        self.orbpop.save_hdf(filename,path='{}/orbpop'.format(path))
        StarPopulation.save_hdf(self,filename,path=path)

class BinaryPopulation_FromH5(BinaryPopulation,StarPopulation_FromH5):
    def __init__(self,filename,path=''):
        """Loads in a BinaryPopulation saved to .h5
        """
        StarPopulation_FromH5.__init__(self,filename,path=path)
        self.orbpop = OrbitPopulation_FromH5(filename,path='{}/orbpop'.format(path))

class BGStarPopulation(StarPopulation):
    def __init__(self,stars,mags=None,maxrad=1800,density=None,name=''):
        self.mags = mags
        if density is None:
            self.density = len(stars)/((3600.*u.arcsec)**2) #default is for TRILEGAL sims to be 1deg^2
        else:
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
        
    def save_hdf(self,filename,path=''):
        properties = {'_maxrad':self._maxrad,
                      'density':self.density}
        StarPopulation.save_hdf(self,filename,path=path,properties=properties)

class BGStarPopulation_FromH5(BGStarPopulation,StarPopulation_FromH5):
    def __init__(self,filename,path=''):
        """Loads in a BGStarPopulation saved to .h5
        """
        StarPopulation_FromH5.__init__(self,filename,path=path)
        store = pd.HDFStore(filename)
        properties = store.get_storer('{}/stars'.format(path)).attrs.properties
        self._maxrad = properties['_maxrad']
        self.density = properties['density']
        store.close()




#methods below should be applied to relevant subclasses
'''
    def set_dmaglim(self,dmaglim):
        if not (hasattr(self,'blendmag') and hasattr(self,'dmaglim')):
            return
        self.dmaglim = dmaglim
        self.apply_constraint(LowerLimit(self.dmags(),self.dmaglim,name='bright blend limit'),overwrite=True)
        self._apply_all_constraints()  #not necessary?
'''

