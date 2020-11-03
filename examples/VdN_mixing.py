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

number_iterations = 1000
LAMBDA_RANGE = [0., 1.] # LAMBDA values imply the fraction of snow in groundwater
# Number of best simulations using which lambda is computed
BEST_SIM_PER = 5. # In percentage

outputfilepath = "OutputFiles/VdN/"


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
lambda_params = np.random.uniform(LAMBDA_RANGE[0], LAMBDA_RANGE[1], number_iterations)

# Assuming constant error variance computed from data
likelihood_std_params_h2 = np.full(number_iterations, np.std(gw_df["H2 isotope"].values, ddof=1))
likelihood_std_params_o18 = np.full(number_iterations, np.std(gw_df["O18 isotope"].values, ddof=1))

# Running HydroMix for H2

rain, snow, gw = rain_df["H2 isotope"].values, snow_df["H2 isotope"].values, gw_df["H2 isotope"].values
likelihood_h2, LAMBDA_H2, ErrorSTD_H2 = hydro_mix(snow, rain, gw, lambda_params, likelihood_std_params_h2, number_iterations)

# Writing output in csv file

final_lis = [["Snow ratio", "Log likelihood", "Error std"]]
path = outputfilepath + "H2_results.csv"
for index in range(0, len(likelihood_h2)):
	final_lis.append([round(LAMBDA_H2[index], 2), round(likelihood_h2[index], 2), round(ErrorSTD_H2[index], 2)])

print (path)

# Running HydroMix for O18

rain, snow, gw = rain_df["O18 isotope"].values, snow_df["O18 isotope"].values, gw_df["O18 isotope"].values
likelihood_o18, lambda_o18, error_std_o18 = hydro_mix(snow, rain, gw, lambda_params, likelihood_std_params_o18, number_iterations)

# Writing output in csv file

final_lis = [["Snow ratio", "Log likelihood", "Error std"]]
path = outputfilepath + "O18_results.csv"
for index in range(0, len(likelihood_o18)):
	final_lis.append([round(lambda_o18[index], 2), round(likelihood_o18[index], 2), round(error_std_o18[index], 2)])

print (path)

# %% Histogram plot showing snow ratio in groundwater using H2 and O18

plt.figure()
plt.hist(LAMBDA_H2[0:int(0.01 * BEST_SIM_PER * number_iterations)], color='blue', alpha=0.4, label=r'$\delta^{2}$H' + u' \u2030 (VSMOW)', normed='True')
plt.hist(lambda_o18[0:int(0.01 * BEST_SIM_PER * number_iterations)], color='red', alpha=0.4, label=r'$\delta^{18}$O' + u' \u2030 (VSMOW)', normed='True')
plt.xlim(0., 1.)
plt.grid(linestyle='dotted')
plt.xlabel("Fraction of snow in groundwater", fontsize=14)
plt.ylabel("Normalised frequency", fontsize=14)
plt.legend()
plt.tight_layout()
path = outputfilepath + "posterior.jpeg"
plt.savefig(path, dpi=300)
plt.close()
print (path)


