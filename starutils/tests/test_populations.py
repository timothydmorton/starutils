from __future__ import print_function, division

from starutils.populations import Raghavan_BinaryPopulation
from starutils.populations import MultipleStarPopulation
from starutils.populations import BGStarPopulation_TRILEGAL
from starutils.populations import ColormatchMultipleStarPopulation

import os

def test_raghavan(filename='/tmp/test_raghavan.h5'):
    pop = Raghavan_BinaryPopulation(1, n=100)
    pop.save_hdf(filename, overwrite=True)
    pop2 = Raghavan_BinaryPopulation().load_hdf(filename)
    os.remove(filename)

def test_multiple(filename='/tmp/test_multiple.h5'):
    pop = MultipleStarPopulation(1, n=100)
    pop.save_hdf(filename, overwrite=True)
    pop2 = MultipleStarPopulation().load_hdf(filename)
    os.remove(filename)

def test_colormatch(filename='/tmp/test_colormatch.h5'):
    mags = {'H': 10.211, 'J': 10.523, 'K': 10.152000000000001} #Kepler-22
    pop = ColormatchMultipleStarPopulation(mags, m1=(1,0.1),
                                           age=(9.7,0.1),
                                           feh=(0,0.1), n=100)
    pop.save_hdf(filename, overwrite=True)
    pop2 = ColormatchMultipleStarPopulation().load_hdf(filename)    
    os.remove(filename)

#def test_bg(filename): 
#    ra, dec = (289.21749900000003, 47.884459999999997) #Kepler-22
#    pop = BGStarPopulation_TRILEGAL('kepler22b.h5', ra, dec)
#    pop.save_hdf('test_bg.h5', overwrite=True)
#    pop2 = BGStarPopulation_TRILEGAL().load_hdf('test_bgpop.h5')    
