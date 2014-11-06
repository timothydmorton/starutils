from __future__ import print_function,division
import numpy as np
import logging

class ConstraintDict(dict):
    def __hash__(self):
        hashint = 0
        for name in self:
            hashint += self[name].__hash__()
        return hashint

class Constraint(object):
    def __init__(self,mask,name='',**kwargs):
        self.name = name
        self.ok = mask
        self.wok = np.where(self.ok)[0]
        self.frac = float(self.ok.sum())/np.size(mask)
        for kw in kwargs:
            setattr(self,kw,kwargs[kw])

    def __eq__(self,other):
        return hash(self) == hash(other)

    def __ne__(self,other):
        return not self.__eq__(other)

    def __hash__(self):
        key = 0
        key += hash(self.name)
        key += hash(self.wok[0:100].__str__())
        key += hash(self.ok.sum())
        return key

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<%s: %s>' % (type(self),str(self))

class JointConstraintAnd(Constraint):
    def __init__(self,c1,c2,name='',**kwargs):
        self.name = name
        mask = ~(~c1.ok & ~c2.ok)
        Constraint.__init__(self,mask,name=name,**kwargs)

class JointConstraintOr(Constraint):
    def __init__(self,c1,c2,name='',**kwargs):
        self.name = name
        mask = ~(~c1.ok | ~c2.ok)
        Constraint.__init__(self,mask,name=name,**kwargs)

class RangeConstraint(Constraint):
    def __init__(self,vals,lo,hi,name='',**kwargs):
        self.lo = lo
        self.hi = hi
        Constraint.__init__(self,(vals > lo) & (vals < hi),
                            name=name,vals=vals,lo=lo,hi=hi,**kwargs)

    def __str__(self,fmt='%.1e'): #implement default string formatting better.....TODO
        return '%s < %s < %s' % (fmt,self.name,fmt) % (self.lo,self.hi)

class UpperLimit(RangeConstraint):
    def __init__(self,vals,hi,name='',**kwargs):
        RangeConstraint.__init__(self,vals,name=name,lo=-np.inf,hi=hi,**kwargs)
        
    def __str__(self,fmt='%.1e'):
        return '%s < %s' % (self.name,fmt) % (self.hi)    

class LowerLimit(RangeConstraint):
    def __init__(self,vals,lo,name='',**kwargs):
        RangeConstraint.__init__(self,vals,name=name,lo=lo,hi=np.inf,**kwargs)

    def __str__(self,fmt='%.1e'):
        return '%s > %s' % (self.name,fmt) % (self.lo)

class MeasurementConstraint(RangeConstraint):
    def __init__(self,vals,val,dval,thresh=3,name='',**kwargs):
        lo = val - thresh*dval
        hi = val + thresh*dval
        RangeConstraint.__init__(self,vals,lo,hi,name=name,val=val,
                                 dval=dval,thresh=thresh,**kwargs)

class FunctionLowerLimit(Constraint):
    def __init__(self,xs,ys,fn,name='',**kwargs):
        Constraint.__init__(self,ys > fn(xs),name=name,xs=xs,ys=ys,fn=fn,**kwargs)
    
class FunctionUpperLimit(Constraint):
    def __init__(self,xs,ys,fn,name='',**kwargs):
        Constraint.__init__(self,ys < fn(xs),name=name,
                            xs=xs,ys=ys,fn=fn,**kwargs)
    
class ContrastCurveConstraint(FunctionLowerLimit):
    def __init__(self,rs,dmags,cc,name='CC',**kwargs):
        self.rs = rs
        self.dmags = dmags
        self.cc = cc
        FunctionLowerLimit.__init__(self,rs,dmags,cc,name=name,**kwargs)
        
    def __str__(self):
        return '%s contrast curve' % self.name

    def update_rs(self,rs):
        self.rs = rs
        FunctionLowerLimit.__init__(self,rs,self.dmags,self.cc,name=self.name)
        logging.info('%s updated with new rsky values.' % self.name)

class VelocityContrastCurveConstraint(FunctionLowerLimit):
    def __init__(self,vels,dmags,vcc,name='VCC',**kwargs):
        self.vels = vels
        self.dmags = dmags
        self.vcc = vcc
        FunctionLowerLimit.__init__(self,vels,dmags,vcc,name=name,**kwargs)
