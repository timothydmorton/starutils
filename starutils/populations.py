from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging

from orbitutils import OrbitPopulation
from plotutils import setfig,plot2dhist

from .constraints import Constraint
from .constraints import ConstraintDict,MeasurementConstraint,RangeConstraint

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

    def save_hdf(self,filename,path=''):
        self.stars.to_hdf(filename,'{}/stars'.format(path))
        self.constraint_df.to_hdf(filename,'{}/constraints'.format(path))

        store = pd.HDFStore(filename)
        attrs = store.get_storer('{}/stars'.format(path)).attrs
        attrs.selectfrac_skip = self.selectfrac_skip
        attrs.distribution_skip = self.distribution_skip
        attrs.name = self.name
        attrs.poptype = type(self)
        store.close()

class StarPopulation_FromH5(StarPopulation):
    def __init__(self,filename,path=''):
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
    def __init__(self,primary,secondary,orbpop=None,
                 period=None,ecc=None,name='',**kwargs):

        """A population of binary stars.

        If ``OrbitPopulation`` provided, that will describe the orbits;
        if not, then orbit population will be generated.

        primary,secondary : ``DataFrame``
            Properties of primary and secondary stars, respectively.
            These get merged into new ``stars`` attribute, with "_A"
            and "_B" tags.

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
        
        if orbpop is None:
            self.orbpop = OrbitPopulation(primary['mass'],
                                          secondary['mass'],
                                          period,ecc)
        else:
            self.orbpop = orbpop

        StarPopulation.__init__(self,stars,name=name)

    @property
    def Rsky(self):
        return self.orbpop.Rsky

    @property
    def RV(self):
        return self.orbpop.RV

#methods below should be applied to relevant subclasses
'''
    def apply_trend_constraint(self,limit,dt):
        """Only works if object has dRV method and plong attribute; limit in km/s"""
        dRVs = np.absolute(self.dRV(dt))
        c1 = UpperLimit(dRVs, limit)
        c2 = LowerLimit(self.stars.Plong, dt*4)

        self.apply_constraint(JointConstraintOr(c1,c2,name='RV monitoring',Ps=self.stars.Plong,dRVs=dRVs))

    def apply_cc(self,cc):
        """Only works if object has rsky, dmags attributes
        """
        rs = self.rsky
        dmags = self.dmags(cc.band)
        self.apply_constraint(ContrastCurveConstraint(rs,dmags,cc,name=cc.name))

    def apply_vcc(self,vcc):
        """only works if has dmags and RV attributes"""
        if type(vcc)==type((1,)):
            vcc = VelocityContrastCurve(*vcc)
        dmags = self.dmags(vcc.band)
        rvs = self.RV
        self.apply_constraint(VelocityContrastCurveConstraint(rvs,dmags,vcc,name='secondary spectrum'))
        
    def set_dmaglim(self,dmaglim):
        if not (hasattr(self,'blendmag') and hasattr(self,'dmaglim')):
            return
        self.dmaglim = dmaglim
        self.apply_constraint(LowerLimit(self.dmags(),self.dmaglim,name='bright blend limit'),overwrite=True)
        self._apply_all_constraints()  #not necessary?
'''

