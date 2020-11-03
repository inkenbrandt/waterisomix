"""
Created on: 27th September 2019 (Harsh Beria)
What it does?

A mixing model to estimate source contribution in a two-source linear system using an average likelihood approach.
Error in estimation is assumed to be normally distributed with zero mean and constant standard deviation (which is computed from data)

Model parameters:
number_iterations => Number of model runs
lambda_params => Fraction of source 1 in the mixture
LIKELIHOOD_std_params => Fixed value

SourceFiles used: NONE

OutputFiles processed:
OutputFiles/Synthetic_exp/

Figures made:
"""


from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
from hydromix.mixingfunctions import *


# HydroMix model parameters (in the given range (inclusive))
NUMBER_ITERATIONS = 1000
LAMBDA_RANGE = [0., 1.]
JUMP_PERCENTAGE = 5. # In percentage (JUMP_PERCENTAGE/2 in both directions)

# Parameters to generate synthetic time series
# Lower and upper bounds of rain and snow isotopic ratio (in H2)
N_SAMPLES = 100 # Number of samples
S1_MEAN, S1_STD, N_S1_SAMPLES = 10., .5, N_SAMPLES
S2_MEAN, S2_STD, N_S2_SAMPLES = 20., .5, N_SAMPLES
N_MIX_SAMPLES = N_SAMPLES

OUTPUTPATH = "OutputFiles/Synthetic_exp/Normal_distribution_0.5Std_" + str(NUMBER_ITERATIONS) + "iter_"
OUTPUTPATH += str(N_SAMPLES) + "sample/"

# %% Setting up a random seed
np.random.seed(1) # Setting up a common seed number

# %% Mixing for lambda ranging from 0.05 to 0.95
LAMBDA = 0.05 # Ratio of source 1 in the mixture
scatterplot_orig_lambda, scatterplot_sim_lambda = [], [] # For displaying in the scatterplot

while (LAMBDA <= 0.96):

	# Computing GW mean and variance
	MIX_MEAN = LAMBDA * S1_MEAN + (1-LAMBDA) * S2_MEAN
	MIX_STD = np.sqrt((LAMBDA*S1_STD)**2 + ((1-LAMBDA)*S2_STD)**2)
	
	# Generating synthetic values for the sources and the mixture
	S1_val = np.random.normal(S1_MEAN, S1_STD, N_S1_SAMPLES)
	S2_val = np.random.normal(S2_MEAN, S2_STD, N_S2_SAMPLES)
	MIX_val = np.random.normal(MIX_MEAN, MIX_STD, N_MIX_SAMPLES)
	
	# Saving it into a csv file
	path = OUTPUTPATH + "input_" + str(LAMBDA) + ".csv"
	output_lis = [["Source 1", "Source 2", "Mixture"]]
	for temp_index in range(len(S1_val)):
		output_lis.append([S1_val[temp_index], S2_val[temp_index], MIX_val[temp_index]])

	print(path)
	
	# List of initial parameter values 
	initParam = [np.random.uniform(LAMBDA_RANGE[0], LAMBDA_RANGE[1])]
	paramLimit = [LAMBDA_RANGE] # Lower and upper limits of the model parameters
	
	# Running the mixing model
	LOGLIKELIHOOD, PARAM, RESIDUAL = hydro_mix_mcmc(S1_val, S2_val, MIX_val, MIX_STD, initParam, paramLimit, NUMBER_ITERATIONS, JUMP_PERCENTAGE)

	mixingRatioLis = [i[0] for i in PARAM]

	# Writing into a csv file
	path = OUTPUTPATH + "output_" + str(LAMBDA) + "_MCMC.csv"
	output_lis = [["Lambda value", "Log likelihood", "Error std", "Residual"]] 
	for index in range(0, len(LOGLIKELIHOOD)):
		output_lis.append([ round(mixingRatioLis[index], 4), round(LOGLIKELIHOOD[index], 4), round(MIX_STD, 4), 
					 round(RESIDUAL[index], 4) ])

	print(path)
	
	
#	scatterplot_orig_lambda.append(LAMBDA)
#	scatterplot_sim_lambda.append(np.median(lambda_params[0:int(best_sim_per * number_iterations / 100.)]))
#	print (LAMBDA, np.median(lambda_params[0:int(best_sim_per * number_iterations / 100.)]))
	LAMBDA += 0.05
	
	
#################################################################################################################################
## %% Plotting scatterplot
#
#plt.figure()
#plt.scatter(scatterplot_orig_lambda, scatterplot_sim_lambda)
#plt.plot([0., 1.], [0., 1.], 'k--', color = 'r')
#plt.xlabel("Original lambda")
#plt.ylabel("Simulated labmda")
#plt.xlim(0., 1.)
#plt.ylim(0., 1.)
#plt.tight_layout()
#path = outputpath + "Figures/scatterplot.jpeg"
#plt.savefig(path)
#
#plt.close()

#################################################################################################################################
