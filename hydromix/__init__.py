try:

    from hydromix.mixingfunctions import *
    from hydromix.watercomp import *
except ImportError:
    from .mixingfunctions import *
    from .watercomp import *

__version__ = '1.0.1'
__author__ = 'Harsh Beria'
__name__ = 'hydromix'

__all__ = ['random_walk','hydro_mix_mcmc','hydro_mix_weighted_mcmc']

