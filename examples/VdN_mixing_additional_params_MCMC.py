"""
Created on: 30th September 2019 (Harsh Beria)
Last updated on
What it does?

Computes the proportion of groundwater that is made of snow vs rain using a MCMC sampler
Introduced additional model parameters (isotopic lapse rate)

SourceFiles used:
SourceFiles/Rain_SP_GW_VdN.xlsx
SourceFiles/Hypsometric_curve_data.xlsx

OutputFiles processed:
OutputFiles/VdN/Lapse_rate/H2_results_MCMC_1000.csv
OutputFiles/VdN/Lapse_rate/O18_results_MCMC_1000.csv

Figures made: NONE
"""
# %% Imports

from __future__ import division
import pandas as pd
import numpy as np
import re
from hydromix.mixingfunctions import *

# %% Main variables

NUMBER_ITERATIONS = 10000

# Bayesian model parameters (in the given range (inclusive))
LAMBDA_RANGE = [0., 1.]  # LAMBDA values imply the fraction of snow in groundwater
O18_LAPSE_SLOPE = [-0.0081, 0.0081]  # Obtained from GNIP Swiss data (-0.0027)
H2_LAPSE_SLOPE = [-0.0582, 0.0582]  # Obtained from GNIP Swiss data (-0.0194)

SWISS_LAPSE = {"H2": -0.0194, "O18": -0.0027}  # Swiss lapse rate according to GNIP data
SWISS_LAPSE_lowBound = {"H2": -0.029, "O18": -0.0039}  # Swiss lapse rate according to GNIP data
SWISS_LAPSE_highBound = {"H2": -0.0076, "O18": -0.0013}  # Swiss lapse rate according to GNIP data

JUMP_PERCENTAGE = 5.  # In percentage (JUMP_PERCENTAGE/2 in both directions)

HYPS_DIC = {}  # Key is elevation and value is the percent of catchment at that elevation

OUTPUTFILEPATH = "OutputFiles/VdN/Lapse_rate/"

# %% Setting up a random seed
np.random.seed(1)  # Setting up a common seed number

# %% Reading the hypsometric curve data

filename = "../../../Downloads/Zenodo_dataset/Zenodo_dataset/SourceFiles/Hypsometric_curve_data.xlsx"
df = pd.read_excel(filename, sheetname='Sheet1')
for index, row in df.iterrows():
    temp_elev_lis = re.findall(r'\d+', row[
        'Elevation band'])  # Identifying the numbers in the string and putting them in a list
    temp_elev_lis = [float(k) for k in temp_elev_lis]  # Converting characters to floats
    elevation_avg = sum(temp_elev_lis) * 1. / len(temp_elev_lis)  # Average elevation in a given elevation band
    HYPS_DIC[elevation_avg] = float(row['Percentage of grids'])

# %% Reading all the isotopic data

filename = "../../../Downloads/Zenodo_dataset/Zenodo_dataset/SourceFiles/Rain_SP_GW_VdN.xlsx"
df = pd.read_excel(filename, sheet_name='Sheet1')

# Separating the isotopic ratios in rain, snow and groundwater
rain_df = df[(df["Type"] == "Rain")]
snow_df = df[(df["Type"] == "Snow")]
gw_df = df[(df["Type"] == "Groundwater")]

#################################################################################################################################
# %% Running HydroMix for H2

rain, snow, gw = rain_df["H2 isotope"].values, snow_df["H2 isotope"].values, gw_df["H2 isotope"].values
rain_elev, snow_elev = rain_df["Elevation (m)"].values, snow_df["Elevation (m)"].values

H2_std = np.std(gw_df["H2 isotope"].values, ddof=1)  # Standard deviation of H2 in groundwater

# List of initial parameter values 
initParam = [np.random.uniform(LAMBDA_RANGE[0], LAMBDA_RANGE[1]),
             np.random.uniform(H2_LAPSE_SLOPE[0], H2_LAPSE_SLOPE[1])]

# Lower and upper limits of the model parameters
paramLimit = [LAMBDA_RANGE, H2_LAPSE_SLOPE]

# Running the mixing model
LOGLIKELIHOOD_H2, PARAM_H2, RESIDUAL_H2 = hydro_mix_mcmc(snow, snow_elev, rain, rain_elev, gw, H2_std,
                                                         initParam, paramLimit, NUMBER_ITERATIONS, HYPS_DIC,
                                                         JUMP_PERCENTAGE)

snowRatioLis_H2 = [i[0] for i in PARAM_H2]
lapseLis_H2 = [i[1] for i in PARAM_H2]

# %% Writing output in csv file

final_lis = [["Snow ratio", "Log likelihood", "Error std", "H2 lapse rate", "Residual"]]
path = OUTPUTFILEPATH + "H2_results_MCMC_" + str(NUMBER_ITERATIONS) + ".csv"
for index in range(0, len(LOGLIKELIHOOD_H2)):
    final_lis.append([round(snowRatioLis_H2[index], 4), round(LOGLIKELIHOOD_H2[index], 4), round(H2_std, 4),
                      round(lapseLis_H2[index], 4), round(RESIDUAL_H2[index], 4)])

print(path)

# %% Running HydroMix for O18

rain, snow, gw = rain_df["O18 isotope"].values, snow_df["O18 isotope"].values, gw_df["O18 isotope"].values
rain_elev, snow_elev = rain_df["Elevation (m)"].values, snow_df["Elevation (m)"].values

O18_std = np.std(gw_df["O18 isotope"].values, ddof=1)  # Standard deviation of O18 in groundwater

# List of initial parameter values 
initParam = [np.random.uniform(LAMBDA_RANGE[0], LAMBDA_RANGE[1]),
             np.random.uniform(O18_LAPSE_SLOPE[0], O18_LAPSE_SLOPE[1])]

# Lower and upper limits of the model parameters
paramLimit = [LAMBDA_RANGE, O18_LAPSE_SLOPE]

# Running the mixing model
LOGLIKELIHOOD_O18, PARAM_O18, RESIDUAL_O18 = hydro_mix_mcmc(snow, snow_elev, rain, rain_elev, gw, O18_std,
                                                            initParam, paramLimit, NUMBER_ITERATIONS, HYPS_DIC,
                                                            JUMP_PERCENTAGE)

snowRatioLis_O18 = [i[0] for i in PARAM_O18]
lapseLis_O18 = [i[1] for i in PARAM_O18]

# %% Writing output in csv file

final_lis = [["Snow ratio", "Log likelihood", "Error std", "O18 lapse rate", "Residual"]]
path = OUTPUTFILEPATH + "O18_results_MCMC_" + str(NUMBER_ITERATIONS) + ".csv"
for index in range(0, len(LOGLIKELIHOOD_O18)):
    final_lis.append([round(snowRatioLis_O18[index], 4), round(LOGLIKELIHOOD_O18[index], 4), round(O18_std, 4),
                      round(lapseLis_O18[index], 4), round(RESIDUAL_O18[index], 4)])

print(path)

##################################################################################################################################
