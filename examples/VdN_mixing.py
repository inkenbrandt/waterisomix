"""
Created on: 8th November 2018 (Harsh Beria)
Last updated on
What it does?

Computes the proportion of groundwater that is made of snow vs rain

SourceFiles used:
SourceFiles/Rain_SP_GW_VdN.xlsx


OutputFiles processed:
OutputFiles/VdN/H2_results.csv
OutputFiles/VdN/O18_results.csv

Figures made:
OutputFiles/VdN/posterior.jpeg
"""

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hydromix.mixingfunctions import *

# Main variables

NUMBER_ITERATIONS = 1000
LAMBDA_RANGE = [0., 1.] # LAMBDA values imply the fraction of snow in groundwater
# Number of best simulations using which lambda is computed
BEST_SIM_PER = 5. # In percentage

OUTPUTFILEPATH = "OutputFiles/VdN/"


# etting up a random seed
np.random.seed(1) # Setting up a common seed number

# Reading all the isotopic data

filename = "../../../Downloads/Zenodo_dataset/Zenodo_dataset/SourceFiles/Rain_SP_GW_VdN.xlsx"
df = pd.read_excel(filename, sheet_name='Sheet1')

# Separating the isotopic ratios in rain, snow and groundwater
rain_df = df[(df["Type"] == "Rain")]
snow_df = df[(df["Type"] == "Snow")]
gw_df = df[(df["Type"] == "Groundwater")]

# Initializing the model parameters
LAMBDA_params = np.random.uniform(LAMBDA_RANGE[0], LAMBDA_RANGE[1], NUMBER_ITERATIONS)

# Assuming constant error variance computed from data
LIKELIHOOD_std_params_H2 = np.full(NUMBER_ITERATIONS, np.std(gw_df["H2 isotope"].values, ddof=1))
LIKELIHOOD_std_params_O18 = np.full(NUMBER_ITERATIONS, np.std(gw_df["O18 isotope"].values, ddof=1))

# Running HydroMix for H2

rain, snow, gw = rain_df["H2 isotope"].values, snow_df["H2 isotope"].values, gw_df["H2 isotope"].values
LIKELIHOOD_H2, LAMBDA_H2, ErrorSTD_H2 = hydro_mix(snow, rain, gw, LAMBDA_params, LIKELIHOOD_std_params_H2, NUMBER_ITERATIONS)

# Writing output in csv file

final_lis = [["Snow ratio", "Log likelihood", "Error std"]]
path = OUTPUTFILEPATH + "H2_results.csv"
for index in range(0, len(LIKELIHOOD_H2)):
	final_lis.append([ round(LAMBDA_H2[index], 2), round(LIKELIHOOD_H2[index], 2), round(ErrorSTD_H2[index], 2) ])

print (path)

# Running HydroMix for O18

rain, snow, gw = rain_df["O18 isotope"].values, snow_df["O18 isotope"].values, gw_df["O18 isotope"].values
LIKELIHOOD_O18, LAMBDA_O18, ErrorSTD_O18 = hydro_mix(snow, rain, gw, LAMBDA_params, LIKELIHOOD_std_params_O18, NUMBER_ITERATIONS)

# Writing output in csv file

final_lis = [["Snow ratio", "Log likelihood", "Error std"]]
path = OUTPUTFILEPATH + "O18_results.csv"
for index in range(0, len(LIKELIHOOD_O18)):
	final_lis.append([ round(LAMBDA_O18[index], 2), round(LIKELIHOOD_O18[index], 2), round(ErrorSTD_O18[index], 2) ])

print (path)

# %% Histogram plot showing snow ratio in groundwater using H2 and O18

plt.figure()
plt.hist(LAMBDA_H2[0:int(0.01 * BEST_SIM_PER * NUMBER_ITERATIONS)], color='blue', alpha=0.4, label=r'$\delta^{2}$H' + u' \u2030 (VSMOW)', normed='True')
plt.hist(LAMBDA_O18[0:int(0.01 * BEST_SIM_PER * NUMBER_ITERATIONS)], color='red', alpha=0.4, label=r'$\delta^{18}$O' + u' \u2030 (VSMOW)', normed='True')
plt.xlim(0., 1.)
plt.grid(linestyle='dotted')
plt.xlabel("Fraction of snow in groundwater", fontsize=14)
plt.ylabel("Normalised frequency", fontsize=14)
plt.legend()
plt.tight_layout()
path = OUTPUTFILEPATH + "posterior.jpeg"
plt.savefig(path, dpi=300)
plt.close()
print (path)


