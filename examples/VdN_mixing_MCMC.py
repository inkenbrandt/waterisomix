"""
Created on: 8th November 2018 (Harsh Beria)
Last updated on: 26th September 2019 (Saved last accepted parameter in rejection step of Metropolis Hastings step)
What it does?

Computes the proportion of groundwater that is made of snow vs rain using a MCMC sampler

SourceFiles used:
SourceFiles/Rain_SP_GW_VdN.xlsx


OutputFiles processed:
OutputFiles/VdN/H2_results_MCMC.csv
OutputFiles/VdN/O18_results_MCMC.csv

Figures made: NONE
"""

from __future__ import division
import pandas as pd
import numpy as np
from hydromix.mixingfunctions import *
# %% Main variables

NUMBER_ITERATIONS = 1000
LAMBDA_RANGE = [0., 1.] # LAMBDA values imply the fraction of snow in groundwater
JUMP_PERCENTAGE = 5. # In percentage (JUMP_PERCENTAGE/2 in both directions)

OUTPUTFILEPATH = "OutputFiles/VdN/"

# %% Setting up a random seed
np.random.seed(1234) # Setting up a common seed number

# %% Reading all the isotopic data

filename = "../../../Downloads/Zenodo_dataset/Zenodo_dataset/SourceFiles/Rain_SP_GW_VdN.xlsx"
df = pd.read_excel(filename, sheet_name='Sheet1')

# Separating the isotopic ratios in rain, snow and groundwater
rain_df = df[(df["Type"] == "Rain")]
snow_df = df[(df["Type"] == "Snow")]
gw_df = df[(df["Type"] == "Groundwater")]

# %% Running HydroMix for H2

rain, snow, gw = rain_df["H2 isotope"].values, snow_df["H2 isotope"].values, gw_df["H2 isotope"].values

H2_std = np.std(gw_df["H2 isotope"].values, ddof=1) # Standard deviation of H2 in groundwater

# List of initial parameter values 
initParam = [np.random.uniform(LAMBDA_RANGE[0], LAMBDA_RANGE[1])]

# Lower and upper limits of the model parameters
paramLimit = [LAMBDA_RANGE]

# Running the mixing model
LOGLIKELIHOOD_H2, PARAM_H2, RESIDUAL_H2 = hydro_mix(snow, rain, gw, H2_std,
												   initParam, paramLimit, NUMBER_ITERATIONS, JUMP_PERCENTAGE)

snowRatioLis_H2 = [i[0] for i in PARAM_H2]

# %% Writing output in csv file

final_lis = [["Snow ratio", "Log likelihood", "Error std", "Residual"]]
path = OUTPUTFILEPATH + "H2_results_MCMC_" + str(NUMBER_ITERATIONS) + ".csv"
for index in range(0, len(LOGLIKELIHOOD_H2)):
	final_lis.append([ round(snowRatioLis_H2[index], 4), round(LOGLIKELIHOOD_H2[index], 4), round(H2_std, 4), 
				   round(RESIDUAL_H2[index], 4) ])


# %% Running HydroMix for O18

rain, snow, gw = rain_df["O18 isotope"].values, snow_df["O18 isotope"].values, gw_df["O18 isotope"].values

O18_std = np.std(gw_df["O18 isotope"].values, ddof=1) # Standard deviation of O18 in groundwater

# List of initial parameter values 
initParam = [np.random.uniform(LAMBDA_RANGE[0], LAMBDA_RANGE[1])]

# Lower and upper limits of the model parameters
paramLimit = [LAMBDA_RANGE]

# Running the mixing model
LOGLIKELIHOOD_O18, PARAM_O18, RESIDUAL_O18 = hydro_mix_mcmc(snow, rain, gw, O18_std, initParam, paramLimit, NUMBER_ITERATIONS, JUMP_PERCENTAGE)

snowRatioLis_O18 = [i[0] for i in PARAM_O18]

# %% Writing output in csv file

final_lis = [["Snow ratio", "Log likelihood", "Error std", "Residual"]]
path = OUTPUTFILEPATH + "O18_results_MCMC_" + str(NUMBER_ITERATIONS) + ".csv"
for index in range(0, len(LOGLIKELIHOOD_O18)):
	final_lis.append([ round(snowRatioLis_O18[index], 4), round(LOGLIKELIHOOD_O18[index], 4), round(O18_std, 4),
				   round(RESIDUAL_O18[index], 4) ])
