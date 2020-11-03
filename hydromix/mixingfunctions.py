from __future__ import division
import numpy as np
import random, datetime, calendar
import pandas as pd
import matplotlib.pyplot as plt


def random_walk(initial_param, param_limit, step):
    """Generates the new parameter set

    Args:
        initial_param (list) : Initial parameter values (LIST)
        param_limit : Lower and upper limits of the list of parameters ([[lambdaLowVal, lambdaHighVal], [StdLowVal, StdHighVal])
        step (float) : Defines the maximum extent of jump from the current parameter state (in %)

    Returns:
        The updated parameter state (LIST)
    """

    stepChange = [(i[1] - i[0]) * 0.005 * step for i in param_limit]  # List of step changes

    # Lower and upper limits for all parameters
    lowerParLimit = [max(initial_param[index] - stepChange[index], param_limit[index][0]) for index in
                     range(len(initial_param))]
    upperParLimit = [min(initial_param[index] + stepChange[index], param_limit[index][1]) for index in
                     range(len(initial_param))]

    # Generating new parameter set by sampling uniformly in the given range
    newParam = [np.random.uniform(lowerParLimit[index], upperParLimit[index]) for index in range(len(initial_param))]

    return newParam  # Returns the list containing new parameters


def hydro_mix_mcmc(source1, source2, mixture, std_dev, param_init, param_limit, nb_iter, random_walk_step=5):
    """Contains the core of HydroMix

	Args:
        source1 : Concentration data of source 1
	    source2 : Concentration data of source 2
	    mixture : Concentration data of the mixture made up of a linear combination of sources 1 and 2
	    std_dev : Standard deviation of the mixture (specified apriori and not calibrated)
	    param_init : Initial value of the list of parameters ([lambdaValue, stdValue])
	    param_limit : Lower and upper limits of the list of parameters ([[lambdaLowVal, lambdaHighVal])
	    nb_iter : Number of MCMC runs
	    random_walk_step : In percentage to define the maximum extent of jump from the current parameter state

	Returns a tuple containing [[LOG LIKELIHOOD VALUES], [PARAMETER VALUES]]

	"""

    logLikelihoodLis = []
    param_lis = [param_init]
    for i in range(nb_iter):

        # Compute new parameter set
        updatedParam = random_walk(param_lis[-1], param_limit, random_walk_step)

        # Log likelihood computation
        lambda_param = updatedParam[0]
        temp_loglikelihood = []
        for temp_mix in mixture:
            temp_value = 0.
            temp_residual = []
            for temp_s1 in source1:
                for temp_s2 in source2:
                    temp_estimated_mix = lambda_param * 1. * temp_s1 + (1 - lambda_param) * temp_s2 * 1.

                    # Log likelihood computation
                    temp_value -= (0.5 * np.log(2 * np.pi * std_dev * std_dev))
                    temp_value -= (0.5 * (temp_estimated_mix - temp_mix) * (temp_estimated_mix - temp_mix)) / (
                            std_dev * std_dev)
                    temp_residual.append(temp_estimated_mix - temp_mix)

            temp_loglikelihood.append(temp_value)
        ll_value = np.sum(temp_loglikelihood)

        # Hastings test
        if i == 0:  # First iteration (accept it)
            logLikelihoodLis.append(ll_value)
        else:
            alpha = np.exp(ll_value - logLikelihoodLis[-1])
            if (alpha > 1) or (np.random.rand() > (1 - alpha)):  # Accept the new move
                param_lis.append(updatedParam)
                logLikelihoodLis.append(ll_value)

        # For displaying purposes
        if i % 100 == 0:
            print("Iteration number:" + str(i + 1) + ", Acceptance: " + str(len(logLikelihoodLis) / (i + 1)))

    return logLikelihoodLis, param_lis


def hydro_mix_weighted_mcmc(source1, source1_weight, source2, source2_weight, mixture, std_dev, param_init, param_limit,
                            nb_iter, random_walk_step=5):
    """Contains the core of HydroMix

    Args:
    	source1 : Concentration data of source 1
	    source1_weight :
	    source2 : Concentration data of source 2
	    source2_weight :
	    mixture : Concentration data of the mixture made up of a linear combination of sources 1 and 2
	    std_dev : Standard deviation of the mixture (specified apriori and not calibrated)
	    param_init : Initial value of the list of parameters ([lambdaValue, stdValue])
	    param_limit : Lower and upper limits of the list of parameters ([[lambdaLowVal, lambdaHighVal])
	    nb_iter : Number of MCMC runs
	    random_walk_step : In percentage to define the maximum extent of jump from the current parameter state

	Returns:
	     a tuple containing [[LOG LIKELIHOOD VALUES], [PARAMETER VALUES]]

	"""

    log_likelihood_lis, param_lis = [], [param_init]
    std_param = std_dev
    for i in range(nb_iter):

        # Compute new parameter set
        updatedParam = random_walk(param_lis[-1], param_limit, random_walk_step)

        # Log likelihood computation
        lambda_param = updatedParam[0]
        temp_loglikelihood = []
        for temp_mix in mixture:
            temp_value = 0.
            temp_residual = []
            for temp_s1, temp_s1Weight in zip(source1, source1_weight):
                for temp_s2, temp_s2Weight in zip(source2, source2_weight):
                    temp_estimated_mix = lambda_param * 1. * temp_s1 + (1 - lambda_param) * temp_s2 * 1.

                    # Weight computation
                    temp_weight = temp_s1Weight * temp_s2Weight

                    # Log likelihood computation
                    temp_value -= temp_weight * (0.5 * np.log(2 * np.pi * std_param * std_param))
                    temp_value -= temp_weight * (
                            0.5 * (temp_estimated_mix - temp_mix) * (temp_estimated_mix - temp_mix)) / (
                                          std_param * std_param)
                    temp_residual.append(temp_estimated_mix - temp_mix)

            temp_loglikelihood.append(temp_value)
        LLValue = np.sum(temp_loglikelihood)

        # Hastings test
        if i == 0:  # First iteration (accept it)
            log_likelihood_lis.append(LLValue)
        else:
            alpha = np.exp(LLValue - log_likelihood_lis[-1])
            if (alpha > 1) or (np.random.rand() > (1 - alpha)):  # Accept the new move
                param_lis.append(updatedParam)
                log_likelihood_lis.append(LLValue)

        # For displaying purposes
        if i % 100 == 0:
            print("Iteration number:" + str(i + 1) + ", Acceptance: " + str(len(log_likelihood_lis) / (i + 1)))
    return log_likelihood_lis, param_lis


def hydro_mix(source1, source2, mixture, lambda_prior, error_std_prior, number_iterations):
    """Contains the core of HydroMix

    Args:
        source1 (list): Concentration data of source 1
        source2 (list): Concentration data of source 2
        mixture (list): Concentration data of the mixture made up of a linear combination of sources 1 and 2
        lambda_prior : prior distribution of the proportion of source1 in the mixture
        error_std_prior : prior distribution of the error standard deviation
        number_iterations (int) : Number of times HydroMix has be run

    Returns:
        A tuple containing log likelihood values, lambda values and error standard deviations for all the model runs

    """

    likelihood = []
    for i in range(0, number_iterations, 1):
        # For displaying purposes
        if i % 100 == 0:
            print("Iteration number:" + str(i + 1))

        # Computing likelihood
        temp_likelihood = []
        for temp_mix in mixture:
            temp_value = 0.
            temp_residual = []
            for temp_s1 in source1:
                for temp_s2 in source2:
                    temp_estimated_mix = lambda_prior[i] * 1. * temp_s1 + (1 - lambda_prior[i]) * temp_s2 * 1.

                    # Likelihood computation
                    temp_value += (-1 * (temp_estimated_mix - temp_mix) ** 2) / (2 * (error_std_prior[i] ** 2))
                    temp_value -= (0.5 * np.log(2 * np.pi * error_std_prior[i] * error_std_prior[i]))
                    temp_residual.append(temp_estimated_mix - temp_mix)

            temp_likelihood.append(temp_value)
        likelihood.append(np.sum(temp_likelihood))

    # Sorting the parameter sets by best runs (highest likelihood values)
    zipped = sorted(zip(likelihood, lambda_prior, error_std_prior), reverse=True)
    likelihood, lambda_prior, error_std_prior = zip(*zipped)

    return likelihood, lambda_prior, error_std_prior


def hydro_mix_weighted(source1, source1_weight, source2, source2_weight, mixture, lambda_prior, error_std_prior,
                       number_iterations):
    """Contains the core of HydroMix

    Args:
        source1 => Concentration data of source 1
        source1_weight =>
        source2 => Concentration data of source 2
        source2_weight =>
        mixture => Concentration data of the mixture made up of a linear combination of sources 1 and 2
        lambda_prior => prior distribution of the proportion of source1 in the mixture
        error_std_prior => prior distribution of the error standard deviation
        number_iterations => Number of times HydroMix has be run

    Returns:
        a tuple containing log likelihood values, lambda values and error standard deviations for all the model runs

    """
    likelihood = []
    for i in range(0, number_iterations, 1):
        # For displaying purposes
        if i % 100 == 0:
            print("Iteration number:" + str(i + 1))

        # Computing likelihood
        temp_likelihood = []
        for temp_mix in mixture:
            temp_value = 0.
            temp_residual = []
            for temp_s1, temp_s1Weight in zip(source1, source1_weight):
                for temp_s2, temp_s2Weight in zip(source2, source2_weight):
                    temp_estimated_mix = lambda_prior[i] * 1. * temp_s1 + (1 - lambda_prior[i]) * temp_s2 * 1.

                    # Weight computation
                    temp_weight = temp_s1Weight * temp_s2Weight

                    # Log likelihood computation
                    temp_value += temp_weight * (-1 * (temp_estimated_mix - temp_mix) ** 2) / (
                            2 * (error_std_prior[i] ** 2))
                    temp_value -= temp_weight * (0.5 * np.log(2 * np.pi * error_std_prior[i] * error_std_prior[i]))
                    temp_residual.append(temp_estimated_mix - temp_mix)

            temp_likelihood.append(temp_value)
        likelihood.append(np.sum(temp_likelihood))

    # Sorting the parameter sets by best runs (highest likelihood values)
    zipped = sorted(zip(likelihood, lambda_prior, error_std_prior), reverse=True)
    likelihood, lambda_prior, error_std_prior = zip(*zipped)

    return likelihood, lambda_prior, error_std_prior


def air_temp_gen(mean_temp=4., ampl_temp=8., years=1., offset=-np.pi / 2.):
    """Generate daily time series of air temperature.
    Assumes a sine wave and adds a normally distributed error with variance = 0.2 * amplitude of temperature

	Args:
    	mean_temp (float) : Mean annual temperature
	    ampl_temp (float) : Amplitude of the sine wave of the temperature time series
	    years (int) : Number of years for which the time series is to be obtained
	    offset (float) : To take into account that January temperature is lowest and July temperature is highest

	Returns:
	    A numpy array containing the time series of airTemp
	"""

    numb_of_days = int(365 * years)
    dayNumb = np.linspace(1, numb_of_days, numb_of_days)  # Corresponding to the number of days in a year
    airTemp = ampl_temp * np.sin(2 * np.pi * dayNumb / 365. + offset) + mean_temp + np.random.normal(loc=0,
                                                                                                     scale=0.2 * ampl_temp,
                                                                                                     size=numb_of_days)
    return (airTemp)


def poisson_prcp(numb_event=30, mean_prcp=1500., years=1.):
    """Generate daily time series of precipitation assuming the time between precipitation events comes from a
    poisson distribution & precipitation amount comes from an exponential distribution

	Args:
        numb_event (int) : Number of precipitation events in a year
	    mean_prcp (float) : Average annual precipitation
	    years (int) : Number of years for which time series is to be obtained

	Returns:
	     a numpy array with daily precipitation values
	"""
    intermittent_times = np.random.poisson(lam=365. / numb_event, size=int(round(years * numb_event)))
    precip_days = np.cumsum(intermittent_times)
    precip_days = precip_days[
        (precip_days > 1) & (precip_days < int(round(years * 365)))]  # Removing dates outside the possible range
    precip_amt = np.random.exponential(scale=mean_prcp / numb_event, size=len(precip_days))

    prcp = np.zeros(int(years * 365))
    for index in range(len(precip_days)):
        prcp[precip_days[index] - 1] = precip_amt[index]
    return (prcp)


def prcp_iso(mean_iso=-80., ampl_iso=40., years=1., offset=-np.pi / 2.):
    """Generate time series of precipitation isotopic ratio.
    Assumes a sine wave and adds a normally distributed error with variance = 0.2 * amplitude of isotopic ratio

	Args:
        mean_iso (float) : Mean isotopic ratio of precipitation
	    ampl_iso (float) : Amplitude of the sine wave of the precipitation isotopic time series
	    years (int) : Number of years for which time series is to be obtained
	    offset (float) : To take into account that January isotopic ratio is the most negative and July isotopic ratio is the least negative

	Returns:
	    a numpy array containing the time series of precipitation isotopic ratio
	"""

    numb_of_days = int(365 * years)
    dayNumb = np.linspace(1, numb_of_days, numb_of_days)  # Corresponding to the number of days in a year
    prcp_iso = ampl_iso * np.sin(2 * np.pi * dayNumb / 365. + offset) + mean_iso + np.random.normal(loc=0,
                                                                                                    scale=0.2 * ampl_iso,
                                                                                                    size=numb_of_days)
    return prcp_iso


def catchment_avg_isotope(isotope_ratio, elev_sample_point, slope_lapse_rate, hypsometric_dic):
    """Computes the catchment averaged isotope ratio from isotope_ratio at a point using the lapse rate
    and hypsometric curve data

    Args:
        isotope_ratio (float) : Ratio of isotope at a given point
        elev_sample_point (float) : Elevation at which the sample was taken and isotope ratio was obtained
        slope_lapse_rate (float) : slope of the lapse rate line
        hypsometric_dic (dict) : Key is elevation and value is the percent of catchment at that elevation

    Returns:
        the catchment averaged isotope ratio
    """

    isotope_dic = [0., 0.]  # First is for sum and second is the normalizing factor

    for elevation in hypsometric_dic:
        estimated_ratio = isotope_ratio + slope_lapse_rate * 1. * (elevation - elev_sample_point)
        isotope_dic[0] += (estimated_ratio * hypsometric_dic[elevation] * 1.)
        isotope_dic[1] += hypsometric_dic[elevation]

    return isotope_dic[0] / isotope_dic[1]
