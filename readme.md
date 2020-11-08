# WaterIsoMix v0.0.1: a mixture of Bayesian mixing scripts for attributing uncertain hydrological sources 
A mixture of scripts taken from `HydroMix` by [Harsh Beria](https://github.com/harshberia93) and `WaterCompare` by [Gabriel Bowen](https://github.com/bumbanian). 

## HydroMix
* `HydroMix` scripts in this library are the unofficial copy of the original downloaded from [the zenodo repository](https://doi.org/10.5281/zenodo.3475429) <br>
* The original `Hydromix` was written by [Harsh Beria](https://github.com/harshberia93). <br>
* HydroMix was originally implemented to solve mixing problems in hydrology, such as 2-component hydrograph separation, resolving sources of tree water uptake, role of snowmelt in groundwater recharge, etc. 
* HydroMix formulates the linear mixing problem in a Bayesian inference framework, where the model error is parameterized instead of the source distributions. This makes the framework suitable for problems where the source sample size is small. Also, it is easy to introduce additional model parameters that can transform the source composition and infer them jointly. For problems with large sample size, HydroMix becomes computationally very expensive. In such cases, parameterizing the source distributions, and then inferring the probability density function of the mixing ratio using a probabilistic programming language (eg: Stan) is more effective.<br></p>
* A detailed manuscript describing the model can be seen at: https://www.geosci-model-dev-discuss.net/gmd-2019-69/<br>

## WaterCompare
* This library also includes scripts translated from Gabe Bowen's R library `WaterCompare`. `Watercompare` uses similar logic as Hydromix. <br>
* The R script can be found at [its github repo](https://github.com/SPATIAL-Lab/watercompare) 
* Scripts for paper on isotopic comparison of water samples including evaporation:
`Bowen, G. J., Putman, A., Brooks, J. R., Bowling, D. R., Oerter, E. J., & Good, S. P. (2018). Inferring the source of evaporated waters using stable H and O isotopes. Oecologia, 187(4), 1025-1039.`
