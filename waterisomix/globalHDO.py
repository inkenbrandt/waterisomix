#!/uufs/chpc.utah.edu/sys/pkg/python/2.7.3_rhel6/etc/python
# -*- coding: utf-8 -*-
from __future__ import division

# ---------------------------------------------------------------------------------------
#
#   Standard Python Import Statements
#

# Imports
import matplotlib

matplotlib.use('Agg')
import numpy as np
import scipy as sp
import matplotlib as plt
import os
import sys
import datetime
import pickle
from scipy.io import netcdf
import scipy.stats as stats
import math
from numpy import matrix
import struct
import time
import pickle
from pylab import *
# from mpl_toolkits.basemap import Basemap
import netCDF4
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import multiprocessing
from scipy.stats.kde import gaussian_kde
import matplotlib

# Options
doInParallel = False  # Run in parallel
doOverOceanOnly = False  # Include land component
bootYears = True  # Resample the bulk flux data

# Constants
rnd_cnt = 1000
snum = 1000
sstart = 0
seaErr = 1.28
sigma_ZS_A = 13.5
sigma_TS_A = 2
sigma_RH_A = 0.1
n_bnds1 = [1e10, 1e10];
t_bnds1 = [.0020, .0060]  # Delta_E_O = -33.03
n_bnds2 = [1e10, 1e10];
t_bnds2 = [.0020, .0125]  # Delta_E_O = -33.03
fracInterceptedMu = 0.2018  # From Wang 2007
fracInterceptedSig = 0.0988  # From Wang 2007
DioD = 0.9757
disTOLL = [500, 12]
numLyrsPBL = 2
gridRez = 2.0
sYear = 2005;
eYear = 2013
scf = (1 / 1000 * 1000.0 ** -3) / 1000  # 10^3 km^3/yr
min_tol = 1e-3
if doOverOceanOnly:
    finalMinTol = np.inf
else:
    finalMinTol = min_tol

num_cores = multiprocessing.cpu_count()
np.seterr(all='ignore')

# Files and directories
CURdir = os.getcwd()
optStr = '%dkms_%dhrs_%dlyrs_%.1frez' % (disTOLL[0], disTOLL[1], numLyrsPBL, gridRez)
pklMapFile = '%s/NPY/%s_maps' % (CURdir, optStr);
simFolder = 'GlobalHDO_SIMS'
if not os.path.exists(simFolder):
    os.makedirs(simFolder)

# Print options
print('\n\n\n * * * START  * * * \n')
print("The Python version is %s.%s.%s" % sys.version_info[:3])
print("The current directory is: %s" % CURdir)
print("Using %d processors" % num_cores)
print('doInParallel = ', doInParallel)
print('doOverOceanOnly = ', doOverOceanOnly)
print('bootYears = ', bootYears)

# ---------------------------------------------------------------------------------------
#
#   Sub-functions
#

# Some Time Stats
timeList = [time.time()]

def doTimeDisp(doDisp):
    timeList.append(time.time())
    timeLength = (timeList[-1] - timeList[-2]) / 60
    if doDisp:
        print('***\ntime is', time.ctime(timeList[-1]))
        print('with %.2f min elapsed (%.2f since last clock)\n***\n' % (
            (timeList[-1] - timeList[0]) / 60, (timeList[-1] - timeList[-2]) / 60))
    return timeLength


# Monthly Reshape Years
def monthlyReshapeYrs(grd):
    return np.reshape(grd, (grd.shape[0] / 12, 12, grd.shape[1], grd.shape[2]))


# Monthly Reshape Mean
def monthlyReshapeMean(grd):
    return stats.nanmean(monthlyReshapeYrs(grd), axis=0)


# get the Intercetpion
def getXX(csim):
    rr = csim[0, 15]
    xx = np.zeros(rr.shape)
    # Spatial Value
    xx = xx + np.abs(np.random.normal(fracInterceptedMu, fracInterceptedSig))
    # Check with runoff
    xx[xx > (1 - rr)] = (1 - rr)[xx > (1 - rr)]
    # return
    return xx


# Bulk Flux Simulations
def gammaRVS(mu, std, cnt):
    if (mu > 0) & (std > 0):
        try:
            return np.random.gamma(shape=(mu ** 2 / std ** 2), scale=(std ** 2 / mu), size=cnt)
        except:
            return np.nan
    else:
        return np.nan


# Sim File Name
def simFileName(s):
    return '%s/%s/globalHDO_sim_%d.npy' % (CURdir, simFolder, s)


# Annual weighting!
def getWeightedAnnualSum(csim, matIn, matWgt):
    # Flux weigth dels
    area_lyr = csim[0, 6]
    area_grd = np.tile(area_lyr, (12, 1, 1))
    # Caluclate it
    matOut = np.nansum((area_grd * matWgt * matIn), axis=0) / np.nansum((area_grd * matWgt), axis=0)
    return matOut


# Returns A random Sample From Gabe Precip
def resamplePrecips(n, grid_pr, pr_wgt):
    # Initilize
    delta_P_T_r = np.zeros((n, 12, 90, 180))
    delta_P_wgts = np.zeros((n, 12))
    # Pre calculate the cDFs
    cdf_lst = np.load(pklMapFile + '___grid_pr_cdfs.npy')
    # Get Random Indexes
    ind_lst = np.zeros((n, 12, 2))
    for r in range(n):
        for m in range(12):
            # Get a random index number
            c_rando = np.random.uniform()
            c_cdf = np.copy(cdf_lst[:, m, 1]);
            c_cdf[c_cdf < c_rando] = 2
            c_cdf_index = [np.argmin(np.abs(c_cdf - c_rando))][0]
            c_index = cdf_lst[:, m, 0][c_cdf_index]
            # Save these layers
            delta_P_T_r[r, m, :, :] = grid_pr[c_index, m, :, :]
            delta_P_wgts[r, m] = pr_wgt[c_index, m]
    return delta_P_T_r, delta_P_wgts


# Equlibrium Fractionation Factors
def getAlphas(temp):
    temp = np.array(temp) + 273.15
    # Constants
    a1_Dw = 24.844
    b1_Dw = -76.248
    c1_Dw = 52.612
    a1_Ow = 1.137
    b1_Ow = -0.4156
    c1_Ow = -2.0667
    a1_Dc = 24.844
    b1_Dc = -76.248
    c1_Dc = 71.912
    a1_Oc = 1.137
    b1_Oc = -0.4156
    c1_Oc = 1.0333
    # Get LNalphas
    LNalphaD = np.zeros(temp.shape)
    LnalphaO = np.zeros(temp.shape)
    LNalphaDw = (a1_Dw * (10 ** 6 * temp ** -2) + b1_Dw * (10 ** 3 * temp ** -1) + c1_Dw)
    LNalphaDc = (a1_Dc * (10 ** 6 * temp ** -2) + b1_Dc * (10 ** 3 * temp ** -1) + c1_Dc)
    LNalphaOw = (a1_Ow * (10 ** 6 * temp ** -2) + b1_Ow * (10 ** 3 * temp ** -1) + c1_Ow)
    LNalphaOc = (a1_Oc * (10 ** 6 * temp ** -2) + b1_Oc * (10 ** 3 * temp ** -1) + c1_Oc)
    # Warm and cold values
    frostPoint = 273.15  # 263.15
    LNalphaD = np.zeros(temp.shape);
    LNalphaO = np.zeros(temp.shape)
    if size(temp) > 1:
        LNalphaD[temp >= frostPoint] = LNalphaDw[temp >= frostPoint]
        LNalphaD[temp < frostPoint] = LNalphaDc[temp < frostPoint]
        LNalphaO[temp >= frostPoint] = LNalphaOw[temp >= frostPoint]
        LNalphaO[temp < frostPoint] = LNalphaOc[temp < frostPoint]
    else:
        if temp >= frostPoint:
            LNalphaD = LNalphaDw;
            LNalphaO = LNalphaOw
        else:
            LNalphaD = LNalphaDc;
            LNalphaO = LNalphaOc

    # Alpha Values
    alphaD = np.exp(LNalphaD / 10 ** 3)
    alphaO = np.exp(LNalphaO / 10 ** 3)
    # Scalar case
    if size(temp) == 1: temp = np.array([temp, 0])

    return alphaD, alphaO


# Return alpha values
def get_alpha_S(TS):
    alphaD, alphaO = getAlphas(TS)
    alphaD[alphaD == 0] = np.nan
    alpha_S = alphaD ** -1
    eps_S = (1 - alpha_S) * 10 ** 3
    return alpha_S, eps_S


# Kenetic Fractionation Factor
def get_alpha_K(n_and_t, RH):
    NNN = n_and_t[0]
    theta = n_and_t[1]
    eps_K = theta * (1 - RH) * (1 - DioD ** NNN) * 10 ** 3
    alpha_K = 1 - eps_K / 10 ** 3
    return alpha_K, eps_K


# Updated CG model
def getDelEvapTho(TS, RH, del_L, del_A, n_and_t):
    # Alpha D values
    alpha_S, eps_S = get_alpha_S(TS)
    alpha_K, eps_K = get_alpha_K(n_and_t, RH)
    # Final Calculation
    del_E = ((del_L * alpha_S - RH * del_A - (eps_S + eps_K)) /
             ((1 - RH) + 10 ** -3 * eps_K))
    # Check values & Return them
    del_E[np.isfinite(del_E) == False] = np.nan
    del_E[del_E > 1000] = np.nan;
    del_E[del_E < -1000] = np.nan
    return del_E


# Get fluxweifted tototals
def getFluxSums(csim):
    area_grd = csim[0, 6]
    flux_P_T_yr = csim[0, 31]
    flux_E_T_yr = csim[0, 32]
    lmsk1 = csim[0, 2]
    omsk1 = csim[0, 3]
    # Get Flux sums
    flux_P_O_sums = np.nansum((area_grd * flux_P_T_yr * lmsk1).ravel()) * scf
    flux_P_L_sums = np.nansum((area_grd * flux_P_T_yr * omsk1).ravel()) * scf
    flux_E_O_sums = np.nansum((area_grd * flux_E_T_yr * lmsk1).ravel()) * scf
    # Runoff
    flux_E_L_sums = (flux_P_O_sums + flux_P_L_sums - flux_E_O_sums)
    flux_R_L_sums = (flux_E_O_sums - flux_P_O_sums)
    # Return this
    return flux_P_O_sums, flux_P_L_sums, flux_E_O_sums, flux_E_L_sums, flux_R_L_sums


# Get the delta Sums!
def getDeltaSums(csim):
    area_grd = csim[0, 6]
    flux_P_T_yr = csim[0, 31]
    flux_E_T_yr = csim[0, 32]
    lmsk1 = csim[0, 2]
    omsk1 = csim[0, 3]
    delta_P_T_yr = csim[0, 34]
    delta_E_O_yr = csim[0, 41]
    # Flux sums
    flux_P_O_sums, flux_P_L_sums, flux_E_O_sums, flux_E_L_sums, flux_R_L_sums = getFluxSums(csim)
    # Sums
    dels_P_O_sums = np.nansum((area_grd * flux_P_T_yr * delta_P_T_yr * lmsk1).ravel()) * scf / flux_P_O_sums
    dels_P_L_sums = np.nansum((area_grd * flux_P_T_yr * delta_P_T_yr * omsk1).ravel()) * scf / flux_P_L_sums
    dels_E_O_sums = np.nansum((area_grd * flux_E_T_yr * delta_E_O_yr * lmsk1).ravel()) * scf / flux_E_O_sums
    dels_E_L_sums = (
                            flux_P_O_sums * dels_P_O_sums + flux_P_L_sums * dels_P_L_sums - flux_E_O_sums * dels_E_O_sums) / flux_E_L_sums
    dels_R_L_sums = (flux_E_O_sums * dels_E_O_sums - flux_P_O_sums * dels_P_O_sums) / flux_R_L_sums
    # return
    return dels_P_O_sums, dels_P_L_sums, dels_E_O_sums, dels_E_L_sums, dels_R_L_sums


# Relative fractions
def getFs(rr, xx, f1, f2):
    f_Q = rr
    f_I = xx
    f_T = f1 * (1 - rr - xx)
    f_E = 1 - f_Q - f_T - f_I
    f_EB = f_E * f2
    f_EM = f_E * (1 - f2)
    f_L = (1 - f_I) - (f_T + f_EB)
    f_ETI = f_EB + f_EM + f_T + f_I
    f_ETI = 1 - rr
    return f_EB, f_EM, f_T, f_I, f_Q, f_L, f_ETI


# Get a soil delta given f
def getDeltaLiquid(csim, f1, f2, cc):
    # A different Delta_E each month!
    omsk1 = csim[0, 3]
    dels_P_L = csim[0, 34] * omsk1
    dels_A_T = csim[0, 10]
    grid_TS = csim[0, 11]
    grid_RH = csim[0, 12]
    n_and_t = csim[0, 17]
    rr = csim[0, 15]
    xx = csim[0, 16]
    # Relative fractions
    f_EB, f_EM, f_T, f_I, f_Q, f_L, f_ETI = getFs(rr, xx, f1, f2)
    # Alpha values
    alpha_S, eps_S = get_alpha_S(grid_TS)
    alpha_K, eps_K = get_alpha_K(n_and_t, grid_RH)
    # Alpha Primes
    alpha_p1_all = alpha_K * grid_RH * (dels_A_T + 1000) / (1 - grid_RH)
    alpha_p2_all = alpha_K * alpha_S / (1 - grid_RH)
    matWgt = csim[0, 8]  # Weight by scalled evaporation flux!
    alpha_p1 = getWeightedAnnualSum(csim, alpha_p1_all, matWgt)
    alpha_p2 = getWeightedAnnualSum(csim, alpha_p2_all, matWgt)
    # Equation from paper
    dels_B_L = (
                       ((1 - f_I - (1 - cc) * f_L) * (dels_P_L + 1000) + f_EB * alpha_p1) /
                       ((cc * f_L + f_T) + f_EB * alpha_p2)
               ) - 1000
    # and the moible waters
    dels_L_L = cc * dels_B_L + (1 - cc) * dels_P_L
    dels_M_L = (
                       ((f_L) * (dels_L_L + 1000) + f_EM * alpha_p1) /
                       ((f_Q) + f_EM * alpha_p2)
               ) - 1000
    # Return
    return dels_B_L, dels_M_L


# Get the global partition
def getGlobalPartition(csim):
    # Get the sums here
    flux_P_O_sums, flux_P_L_sums, flux_E_O_sums, flux_E_L_sums, flux_R_L_sums = getFluxSums(csim)
    dels_P_O_sums, dels_P_L_sums, dels_E_O_sums, dels_E_L_sums, dels_R_L_sums = getDeltaSums(csim)
    # Basic Fluxes
    omsk1 = csim[0, 3]
    area_lyr = csim[0, 6]
    area_grd = area_lyr
    flux_P_T_yr = csim[0, 31]
    dels_P_L_yr = csim[0, 34] * omsk1
    dels_A_T_yr = csim[0, 35]
    grid_TS_yr = csim[0, 36]
    grid_RH_yr = csim[0, 37]
    rr = csim[0, 15]
    xx = csim[0, 16]
    n_and_t_land = csim[0, 17]
    dels_A_T = csim[0, 10]
    grid_TS = csim[0, 11]
    grid_RH = csim[0, 12]

    # Returns the delta values
    def getDELS(fffs):
        # Actual fractions
        f1 = fffs[0];
        f2 = fffs[1];
        cc = csim[0, 22]
        f_EB, f_EM, f_T, f_I, f_Q, f_L, f_ETI = getFs(rr, xx, f1, f2)
        # Delta values
        dels_B_L, dels_M_L = getDeltaLiquid(csim, f1, f2, cc)
        dels_TB_L = dels_B_L
        dels_Q_L = dels_M_L
        dels_L_L = cc * dels_B_L + (1 - cc) * dels_P_L_yr
        # Same delta E each month!
        dels_EB_L_all = getDelEvapTho(grid_TS, grid_RH, dels_B_L, dels_A_T, n_and_t_land)
        dels_EM_L_all = getDelEvapTho(grid_TS, grid_RH, dels_M_L, dels_A_T, n_and_t_land)
        matWgt = csim[0, 8]  # Weight by scalled evaporation flux!
        dels_EB_L = getWeightedAnnualSum(csim, dels_EB_L_all, matWgt)
        dels_EM_L = getWeightedAnnualSum(csim, dels_EM_L_all, matWgt)

        # Curent estimate of ET!
        dels_ETI_L = (f_T * dels_TB_L + f_EB * dels_EB_L + f_EM * dels_EM_L + f_I * dels_P_L_yr) / (
                f_T + f_EB + f_EM + f_I)
        # Curent estimate of Runoff!
        dels_R_L_curs = np.nansum((area_lyr * f_Q * flux_P_T_yr * dels_Q_L * omsk1).ravel()) / np.nansum(
            (area_lyr * f_Q * flux_P_T_yr * omsk1).ravel())
        dels_ETI_L_curs = np.nansum((area_lyr * f_ETI * flux_P_T_yr * dels_ETI_L * omsk1).ravel()) / np.nansum(
            (area_lyr * f_ETI * flux_P_T_yr * omsk1).ravel())
        dels_TB_L_curs = np.nansum((area_lyr * f_T * flux_P_T_yr * dels_TB_L * omsk1).ravel()) / np.nansum(
            (area_lyr * f_T * flux_P_T_yr * omsk1).ravel())
        dels_EB_L_curs = np.nansum((area_lyr * f_EB * flux_P_T_yr * dels_EB_L * omsk1).ravel()) / np.nansum(
            (area_lyr * f_EB * flux_P_T_yr * omsk1).ravel())
        dels_EM_L_curs = np.nansum((area_lyr * f_EM * flux_P_T_yr * dels_EM_L * omsk1).ravel()) / np.nansum(
            (area_lyr * f_EM * flux_P_T_yr * omsk1).ravel())
        dels_II_L_curs = np.nansum((area_lyr * f_I * flux_P_T_yr * dels_P_L_yr * omsk1).ravel()) / np.nansum(
            (area_lyr * f_I * flux_P_T_yr * omsk1).ravel())
        # Return stuff
        dels_list = [dels_ETI_L_curs, dels_R_L_curs,
                     dels_TB_L_curs, dels_EB_L_curs, dels_EM_L_curs, dels_II_L_curs]
        dels_maps = [dels_P_L_yr, dels_Q_L, dels_TB_L, dels_EB_L, dels_EM_L, dels_ETI_L]
        return dels_list, dels_maps

    # Minimization function
    def getNewFC(fffs):
        dels_list, dels_maps = getDELS(fffs)
        # Estimate the error
        mse_error = (((dels_E_L_sums - dels_list[0]) ** 2
                      + (dels_R_L_sums - dels_list[1]) ** 2) / 2)
        # Return
        return mse_error

    # Do minimization
    fffs_start = [np.random.uniform(), np.random.uniform()]
    bnds = ((0, 1), (0, 1))
    if doOverOceanOnly == False:
        res = minimize(getNewFC, fffs_start, method='Nelder-Mead', options={'disp': True})
    else:
        res = minimize(getNewFC, fffs_start, method='Nelder-Mead', options={'maxiter': 1, })
    dels_list, dels_maps = getDELS(res.x)
    # Save the flux maps!
    csim[0, 45] = dels_maps[0];
    csim[1, 45] = 'dels_P_L grid'
    csim[0, 46] = dels_maps[1];
    csim[1, 46] = 'dels_Q_L grid'
    csim[0, 47] = dels_maps[2];
    csim[1, 47] = 'dels_TB_L grid'
    csim[0, 48] = dels_maps[3];
    csim[1, 48] = 'dels_EB_L grid'
    csim[0, 49] = dels_maps[4];
    csim[1, 49] = 'dels_EM_L grid'
    csim[0, 50] = dels_maps[5];
    csim[1, 50] = 'dels_ETI_L grid'
    # Return Vars
    return res, dels_list


# ---------------------------------------------------------------------------------------
#
#   Load Some Data Files
#
print
'\n\n\n * * * LOAD DATA * * * \n'
grids_gpcp = np.load(pklMapFile + '___grid_gpcp.npy');
print
'grid_gpcp loaded'
grids_OAFLX = np.load(pklMapFile + '___grid_oaflux.npy');
print
'grid_oaflux loaded'
lmsk1 = np.load(pklMapFile + '___lmsk1.npy');
print
'lmsk1 loaded'
omsk1 = np.load(pklMapFile + '___omsk1.npy');
print
'omsk1 loaded'
lmsk2 = np.load(pklMapFile + '___lmsk2.npy');
print
'lmsk2 loaded'
omsk2 = np.load(pklMapFile + '___omsk2.npy');
print
'omsk2 loaded'
lFrac = np.load(pklMapFile + '___landfrac.npy');
print
'lFrac loaded'
grids_evp_mu = np.load(pklMapFile + '___merra_evp_mu.npy');
print
'grid_merra mean loaded'
grids_evp_sd = np.load(pklMapFile + '___merra_evp_sd.npy');
print
'grid_merra std loaded'
grid_cs = np.load(pklMapFile + '___grid_cs.npy');
print
'grid_cs loaded'
grid_cs = np.reshape(grid_cs, (grid_cs.shape[0] / 12, 12, grid_cs.shape[1], grid_cs.shape[2]))
grid_zs = np.load(pklMapFile + '___grid_ZS_patched.npy');
print
'grid_zs loaded'
grid_RH = np.load(pklMapFile + '___grid_RH_patched.npy');
print
'grid_RH loaded'
grid_TS = np.load(pklMapFile + '___grid_TS_patched.npy');
print
'grid_TS loaded'
grid_x = np.load(pklMapFile + '___grid_x.npy');
print
'grid_x loaded'
grid_y = np.load(pklMapFile + '___grid_y.npy');
print
'grid_y loaded'
grid_pr = np.load(pklMapFile + '___grid_pr_gabe.npy');
print
'grid_pr loaded'
pr_wgts = np.load(pklMapFile + '___grid_wt_gabe.npy');
print
'grid_pr loaded'
grid_area = np.load(pklMapFile + '___grid_areas.npy');
print
'grid_area loaded'
doTimeDisp(True)

# ---------------------------------------------------------------------------------------
#
#   Prepare Data Files
#
print
'\n\n\n * * * PREP DATA * * * \n'
# Use GPCP and OAfluxs
grids_PREC = grids_gpcp[0, :, :, :].squeeze()
grids_PERR = grids_gpcp[1, :, :, :].squeeze()
grids_EVAP = grids_OAFLX[0, :, :, :].squeeze()
grids_EERR = grids_OAFLX[1, :, :, :].squeeze()
grids_EVAP[(grids_EVAP == 0) | (grids_EERR == 0)] = np.nan
grids_EERR[(grids_EVAP == 0) | (grids_EERR == 0)] = np.nan
# Average stats for bulk fluxes
print
'Making monthly averages of Precip'
avgPreT = monthlyReshapeMean(grids_PREC)
sigPreT = monthlyReshapeMean(grids_PERR) / np.sqrt(8)
print
'Making monthly averages of ocean Evap'
avgEvpO = monthlyReshapeMean(grids_EVAP) * lmsk2  # Use the evaporation mask!
sigEvpO = monthlyReshapeMean(grids_EERR) / np.sqrt(8) * lmsk2
print
'Making monthly averages of land Evap'
avgEvpL = monthlyReshapeMean(grids_evp_mu) * omsk2  # Use the evaporation mask!
avgEvpL[avgEvpL < 0] = 0
sigEvpL = monthlyReshapeMean(grids_evp_sd) / np.sqrt(8) * omsk2
# Total Evporation
avgEvpT = np.copy(avgEvpO)
sigEvpT = np.copy(sigEvpO)
avgEvpT[avgEvpT != avgEvpT] = avgEvpL[avgEvpT != avgEvpT]
sigEvpT[sigEvpT != sigEvpT] = sigEvpL[sigEvpT != sigEvpT]

# All Years for booting
print
'Reshaping grids'
grids_PREC_all = monthlyReshapeYrs(grids_PREC)
grids_PERR_all = monthlyReshapeYrs(grids_PERR)
grids_EVAP_all = monthlyReshapeYrs(grids_EVAP)
grids_EERR_all = monthlyReshapeYrs(grids_EERR)
grids_EVAP_all_M = monthlyReshapeYrs(grids_evp_mu)
grids_EERR_all_M = monthlyReshapeYrs(grids_evp_sd)
grids_EVAP_all[grids_EVAP_all < 0] = 0;
grids_EVAP_all = grids_EVAP_all * lmsk2
grids_EVAP_all[grids_EVAP_all != grids_EVAP_all] = grids_EVAP_all_M[grids_EVAP_all != grids_EVAP_all]
grids_EERR_all[grids_EERR_all != grids_EERR_all] = grids_EERR_all_M[grids_EERR_all != grids_EERR_all]
doTimeDisp(True)


# ---------------------------------------------------------------------------------------
#
#   Simulation sub-functions using loaded data
#


# Makes a blank siumlations
def addSims(s):
    # current sims:
    cdat = [None] * 100
    chdr = [None] * 100
    # Append and save
    cdat[0] = grid_x;
    chdr[0] = 'grid_x'
    cdat[1] = grid_y;
    chdr[1] = 'grid_y'
    cdat[2] = lmsk1;
    chdr[2] = 'lmsk1'
    cdat[3] = omsk1;
    chdr[3] = 'omsk1'
    cdat[4] = lmsk2;
    chdr[4] = 'lmsk2'
    cdat[5] = omsk2;
    chdr[5] = 'omsk2'
    cdat[6] = grid_area;
    chdr[6] = 'grid_area'
    cdat[40] = lFrac;
    chdr[40] = 'landFrac'
    cdat[21] = np.inf;
    chdr[21] = 'Inital rmse'
    cdat[51] = np.inf;
    chdr[51] = 'Bootstraped year'
    # mak into array and save it
    csim = np.array([cdat, chdr]);
    np.save(simFileName(s), np.asarray(csim))


# Add bulk fluxes to the simulations
def addFluxes(s):
    # Empty arrays
    grid_P_T = np.zeros((12, 90, 180)) * np.nan
    grid_E_T = np.zeros((12, 90, 180)) * np.nan
    # load old sim
    csim = np.load(simFileName(s))
    area_lyr = csim[0, 6]
    area_grd = np.tile(area_lyr, (12, 1, 1))
    grid_x = csim[0, 0]
    grid_y = csim[0, 1]
    lmsk1 = csim[0, 2]
    omsk1 = csim[0, 3]
    lmsk2 = csim[0, 4]
    omsk2 = csim[0, 5]
    lFrac = csim[0, 40]
    csim[0, 51] = np.random.randint(0, 8);
    # Get random fluxes
    if bootYears == False:
        for cm in range(12):
            for cx in range(grid_x.shape[1]):
                for cy in range(grid_y.shape[0]):
                    grid_P_T[cm, cy, cx] = gammaRVS(avgPreT[cm, cy, cx], sigPreT[cm, cy, cx], 1)
                    grid_E_T[cm, cy, cx] = gammaRVS(avgEvpT[cm, cy, cx], sigEvpT[cm, cy, cx], 1)
    else:
        cBoot = csim[0, 51]
        for cm in range(12):
            for cx in range(grid_x.shape[1]):
                for cy in range(grid_y.shape[0]):
                    grid_P_T[cm, cy, cx] = gammaRVS(grids_PREC_all[cBoot, cm, cy, cx],
                                                    grids_PERR_all[cBoot, cm, cy, cx], 1)
                    grid_E_T[cm, cy, cx] = gammaRVS(grids_EVAP_all[cBoot, cm, cy, cx],
                                                    grids_EERR_all[cBoot, cm, cy, cx], 1)
    # Gap Fill
    grid_P_T[grid_P_T < 0] = np.nan
    grid_E_T[grid_E_T < 0] = np.nan
    grid_P_T[grid_P_T != grid_P_T] = avgPreT[grid_P_T != grid_P_T]
    grid_E_T[grid_E_T != grid_E_T] = avgEvpT[grid_E_T != grid_E_T]
    grid_E_T[grid_E_T < 0] = 0
    grid_P_T[grid_P_T < 0] = 0
    # Append and save
    csim[0, 7] = grid_P_T;
    csim[1, 7] = 'grid_P_T'
    csim[0, 8] = grid_E_T;
    csim[1, 8] = 'grid_E_T'
    grid_E_T_raw = np.copy(grid_E_T)
    # Make Annual Sums
    grid_P_T_yr = np.nansum(grid_P_T, axis=0);
    grid_E_T_yr = np.nansum(grid_E_T, axis=0);
    csim[0, 31] = grid_P_T_yr;
    csim[1, 31] = 'grid_P_T_yr'
    csim[0, 32] = grid_E_T_yr;
    csim[1, 32] = 'grid_E_T_yr'
    rr_old = (grid_P_T_yr - grid_E_T_yr) / grid_P_T_yr * omsk1
    # What is the error
    flux_P_O_sums, flux_P_L_sums, flux_E_O_sums, flux_E_L_sums, flux_R_L_sums = getFluxSums(csim)

    # Function that gets a new Evap grid based on the error function
    def getNewETgrid(scale_factor):
        grid_E_T_scl = np.copy(grid_E_T)
        scaler_under = np.tile(grid_P_T_yr / grid_E_T_yr, (12, 1, 1))
        grid_E_T_scl[(scaler_under < 1) & (omsk2 == 1)] = (grid_E_T_scl * scaler_under)[
            (scaler_under < 1) & (omsk2 == 1)]
        grid_E_T_scl[(scaler_under > 1) & (omsk2 == 1)] = (grid_E_T_scl * scale_factor)[
            (scaler_under > 1) & (omsk2 == 1)]
        scaler_under2 = np.tile(grid_P_T_yr / np.nansum(grid_E_T_scl, axis=0), (12, 1, 1))
        grid_E_T_scl[(scaler_under2 < 1) & (omsk2 == 1)] = (grid_E_T_scl * scaler_under2)[
            (scaler_under2 < 1) & (omsk2 == 1)]
        return grid_E_T_scl

    # Checks an evap error and retruns the RMSE
    def reScaleET(scale_factor):
        grid_E_T_scl = getNewETgrid(scale_factor)
        flux_E_L_curs = np.nansum((area_grd * grid_E_T_scl * omsk2).ravel()) * scf
        evError = flux_E_L_curs / flux_E_L_sums
        return (1 - evError) ** 2

    # Scale the E_T_yr
    res = minimize_scalar(reScaleET)
    grid_E_T_scl = getNewETgrid(res.x)
    grid_E_T_yr_scl = np.nansum(grid_E_T_scl, axis=0)
    # Annual Runoff Ratio!
    rr_new = (grid_P_T_yr - np.nansum(grid_E_T_scl, axis=0)) / grid_P_T_yr * omsk1
    # Save it
    csim[0, 15] = rr_new;
    csim[1, 15] = ' Q/P (adjusted runoff ratio)'
    csim[0, 8] = grid_E_T_scl;
    csim[1, 8] = 'grid_E_T (scaled)'
    csim[0, 32] = grid_E_T_yr_scl;
    csim[1, 32] = 'grid_E_T_yr (scaled)'
    csim[0, 38] = grid_E_T_raw;
    csim[1, 38] = 'raw land evaporaiton'
    csim[0, 39] = res.x;
    csim[1, 39] = 'runoff scaling factor'
    # Save and Continue
    np.save(simFileName(s), csim)


# Add bulk fluxes to the simulations
def addDelta_P_T(s):
    # load Simulations
    csim = np.load(simFileName(s))
    # get fluxes
    dels_P_T_scr, delta_P_T_wgt = resamplePrecips(1, grid_pr, pr_wgts)
    dels_P_T = dels_P_T_scr[0, :, :]
    csim[0, 9] = dels_P_T;
    csim[1, 9] = 'dels_P_T'
    # Do weigthing
    matWgt = csim[0, 7]  # Weight by precip!
    dels_P_T_yr = getWeightedAnnualSum(csim, dels_P_T, matWgt)
    csim[0, 34] = dels_P_T_yr;
    csim[1, 34] = 'dels_P_T_yr'
    # Save and Continue
    np.save(simFileName(s), csim)


# Add PBL met and delta values
def addDeltaAandMet(s):
    # Empty arrays
    c_grid_TS = np.zeros((12, 90, 180)) * np.nan
    c_grid_RH = np.zeros((12, 90, 180)) * np.nan
    c_grid_AA = np.zeros((12, 90, 180)) * np.nan
    c_grid_SEA = np.zeros((12, 90, 180)) * np.nan
    # load old sim
    csim = np.load(simFileName(s))
    # Do Ocean Evaporation
    for m in np.arange(12):
        # Random TS, RH, and ZS
        crn_evp_rnd = np.floor(np.random.uniform(size=3) * rnd_cnt).astype('int')
        sim_grid_zs = grid_zs[crn_evp_rnd[0], m, :, :]
        sim_grid_TS = grid_TS[crn_evp_rnd[1], m, :, :]
        sim_grid_RH = grid_RH[crn_evp_rnd[2], m, :, :]
        sim_grid_SEA = np.copy(sim_grid_RH) * 0
        # Save c sim
        c_grid_AA[m, :, :] = sim_grid_zs
        c_grid_TS[m, :, :] = sim_grid_TS
        c_grid_RH[m, :, :] = sim_grid_RH
        c_grid_SEA[m, :, :] = sim_grid_SEA
    # Add it to it
    csim[0, 10] = c_grid_AA;
    csim[1, 10] = 'delta_A'
    csim[0, 11] = c_grid_TS;
    csim[1, 11] = 'met_TS'
    csim[0, 12] = c_grid_RH;
    csim[1, 12] = 'met_RH'
    csim[0, 13] = c_grid_SEA;
    csim[1, 13] = 'delta_L_O'
    # Add Yearly Weighted sums
    matWgt = csim[0, 8]  # Weight by evap!
    csim[0, 35] = getWeightedAnnualSum(csim, c_grid_AA, matWgt);
    csim[1, 35] = 'delta_A_yr'
    csim[0, 36] = getWeightedAnnualSum(csim, c_grid_TS, matWgt);
    csim[1, 36] = 'met_TS_yr'
    csim[0, 37] = getWeightedAnnualSum(csim, c_grid_RH, matWgt);
    csim[1, 37] = 'met_RH_yr'
    # Save and Continue
    np.save(simFileName(s), csim)


# Add ocean Evaporaiton Flux
def addDelta_E_O(s):
    # Empty list
    delta_E_O_list = []
    # load old sim
    csim = np.load(simFileName(s))
    c_delta_E_O = np.zeros((12, 90, 180)) * np.nan
    # n and t
    c_n_and_t = [np.random.uniform(n_bnds1[0], n_bnds1[1]), np.random.uniform(t_bnds1[0], t_bnds1[1])]
    n_and_t = [np.ones((90, 180)) * c_n_and_t[0], np.ones((90, 180)) * c_n_and_t[1]]
    for m in np.arange(12):
        # Random TS, RH, and ZS
        c_grid_TS = csim[0, 11][m, :, :]
        c_grid_RH = csim[0, 12][m, :, :]
        c_grid_L = csim[0, 13][m, :, :]
        c_grid_A = csim[0, 10][m, :, :]
        c_delta_E_O[m, :, :] = getDelEvapTho(c_grid_TS, c_grid_RH, c_grid_L, c_grid_A, n_and_t)
    # Add and Save
    csim[0, 14] = c_delta_E_O;
    csim[1, 14] = 'delta_E_O'
    csim[0, 19] = c_n_and_t;
    csim[1, 19] = 'n_and_t_ocean'
    # Weighted delta_E
    matWgt = csim[0, 8]  # Weight by precip!
    c_delta_E_O_yr = getWeightedAnnualSum(csim, c_delta_E_O, matWgt)
    csim[0, 41] = c_delta_E_O_yr;
    csim[1, 41] = 'delta_E_O_yr'
    # Save csim
    np.save(simFileName(s), csim)


# Land Isotope Stats
def addLandFluxes(s, itr):
    # load old sim
    csim = np.load(simFileName(s))
    # Calculate Interception Fraction
    rr = csim[0, 15]
    xx = getXX(csim)
    csim[0, 16] = xx;
    csim[1, 16] = ' I/P (intercepted fraction)'
    # Calculate Runoff Ratio
    n_and_t_land = [np.random.uniform(n_bnds2[0], n_bnds2[1]),
                    np.random.uniform(t_bnds2[0], t_bnds2[1])]
    csim[0, 17] = n_and_t_land;
    csim[1, 17] = 'n_and_t_land'
    # Get Delta_L_L
    csim[0, 22] = np.random.uniform();
    csim[1, 22] = 'cc value!'
    # Do the Partitioning
    min_res, dels_list = getGlobalPartition(csim)
    # Check the cMSE
    rmse = min_res.fun ** 0.5
    if (min_res.x[0] < 0) | (min_res.x[0] > 1): rmse = rmse + 10
    if (min_res.x[1] < 0) | (min_res.x[1] > 1): rmse = rmse + 10
    if doOverOceanOnly:    rmse = 0
    # Save the stuff!
    csim[0, 18] = min_res.x[0];
    csim[1, 18] = 'Final f1'
    csim[0, 20] = min_res.x[1];
    csim[1, 20] = 'Final f2'
    csim[0, 21] = rmse;
    csim[1, 21] = 'Final rmse'
    csim[0, 23] = dels_list[0];
    csim[1, 23] = 'dels_ETI_L_curs'
    csim[0, 24] = dels_list[1];
    csim[1, 24] = 'dels_R_L_curs'
    csim[0, 25] = dels_list[2];
    csim[1, 25] = 'dels_TB_L_curs'
    csim[0, 26] = dels_list[3];
    csim[1, 26] = 'dels_EB_L_curs'
    csim[0, 27] = dels_list[4];
    csim[1, 27] = 'dels_EM_L_curs'
    csim[0, 28] = dels_list[5];
    csim[1, 28] = 'dels_II_L_curs'
    csim[0, 29] = itr;
    csim[1, 29] = 'Iterations'
    # Spatial Fields
    f_EB, f_EM, f_T, f_I, f_Q, f_L, f_ETI = getFs(rr, xx, min_res.x[0], min_res.x[1])
    csim[0, 42] = f_T;
    csim[1, 42] = ' T/P'
    csim[0, 43] = f_EB;
    csim[1, 43] = 'EB/P'
    csim[0, 44] = f_EM;
    csim[1, 44] = 'EM/P'
    # Other stuff
    timeLength = doTimeDisp(False)
    if itr == 1:
        csim[0, 30] = timeLength;
        csim[1, 30] = 'timeLength'
    else:
        csim[0, 30] = csim[0, 30] + timeLength
    np.save(simFileName(s), csim)
    print
    '\nSIM %d - ITR %d' % (s, csim[0, 29])
    print
    '\t T/ET=%.3f, Es/E=%.3f, cc=%.3f  (rmse=%.5f)' % (csim[0, 18], csim[0, 20], csim[0, 22], csim[0, 21])
    print
    '\t Tot time %.2f, avg time %.2f' % (csim[0, 30], csim[0, 30] / csim[0, 29])
    # return
    return rmse


# Evaluates a simulation!
def doAsim(s):
    print
    'sim %d' % s
    np.random.seed()
    cITR = 0
    # Make sim
    try:
        csim = np.load(simFileName(s))
        cMSE = csim[0, 21]
    except:
        addSims(s)
        csim = np.load(simFileName(s))
        cMSE = csim[0, 21]
    # Try to make it
    while cMSE > min_tol:
        cITR = cITR + 1
        addFluxes(s)
        addDelta_P_T(s)
        addDeltaAandMet(s)
        addDelta_E_O(s)
        # Get MSE
        cMSE = addLandFluxes(s, cITR)
    doTimeDisp(True)


# ---------------------------------------------------------------------------------------
#
#   Run Model either in parallel or not
#
if __name__ == '__main__':

    print('\n\n\n * * * MAKE SIMS * * * \n')
    if doInParallel:
        if __name__ == "__main__":
            pool = multiprocessing.Pool(processes=num_cores)
            pool.map(doAsim, range(snum))
            pool.close()
            pool.join()
            print('done')
    else:
        for ss in range(snum): doAsim(ss)

    # Finish
    print('\n\n\n * * * ALL DONE * * * \n')
    quit()
