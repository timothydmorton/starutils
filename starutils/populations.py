from __future__ import print_function, division

from orbitutils import OrbitPopulation

class StarPopulation(object):
    def __init__(self,stars,constraints=None,selectfrac_skip=None,distribution_skip=None,
                 **kwargs):
        """A population of stars.
        """
        
