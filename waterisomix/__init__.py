try:

    from waterisomix.mixingfunctions import *
    from waterisomix.watercomp import *
    from waterisomix.hydrocalculator import *
except ImportError:
    from .hydrocalculator import *
    from .mixingfunctions import *
    from .watercomp import *

__version__ = '0.2.0'
__author__ = 'Paul Inkenbrandt'
__name__ = 'waterisomix'

__all__ = ['random_walk','hydro_mix_mcmc','hydro_mix_weighted_mcmc','CraigGordonModel']

