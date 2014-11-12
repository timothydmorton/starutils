from __future__ import print_function,division

import numpy as np
import numpy.random as rand


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
