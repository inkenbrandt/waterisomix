# Based on original R scripts by Gabe Bowen; Found at https://github.com/SPATIAL-Lab/watercompare
#####
# --- preliminaries ----
#####

# import PyMC3
# import statsmodels
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import dirichlet
from scipy.stats import norm


#####functions here

#####
# --- isotope value data frame ----
# creates object of 'iso' data structure used to pass values to functions
#####

#
def iso(H, O, Hsd, Osd, HOc):
    """takes values of H and O isotope composition, SD of each, and covariance
    Args:
        H: mean of d2H for specified source
        O: mean of d18O for specified source
        Hsd: d2H standard deviation
        Osd: d18O standard deviation
        HOc: covariance of O and H

    Returns:
        DataFrame of input variables
    """
    varibs = {'H': H, 'O': O, 'Hsd': Hsd, 'Osd': Osd, 'HOc': HOc}
    for key in varibs.keys():
        if type(varibs[key]) == np.ndarray or type(varibs[key]) == list:
            pass
        else:
            varibs[key] = np.array(varibs[key])

    if np.size(varibs['H']) == np.size(varibs['O']) == np.size(varibs['Hsd']) == np.size(varibs['HOc']):
        data = []
        if np.size(varibs['H']) > 1:
            for i in range(np.size(varibs['H'])):
                print(np.size(varibs['H']))
                data.append([varibs['H'][i], varibs['O'][i],
                             varibs['Hsd'][i], varibs['Osd'][i],
                             varibs['HOc'][i]])
        else:
            data.append([varibs['H'], varibs['O'],
                         varibs['Hsd'], varibs['Osd'],
                         varibs['HOc']])
        return pd.DataFrame(data, columns=varibs.keys())
    else:
        print('lengths of data do not match')


#####
# --- single source implementation ----
#####

def rmvnormapp(obs, ngens=1):
    mean = [obs['H'], obs['O']]
    sigma = [[obs['Hsd'] ** 2, obs['HOc'] * obs['Hsd'] * obs['Osd']],
             [obs['HOc'] * obs['Hsd'] * obs['Osd'], obs['Osd'] ** 2]]

    s = np.random.multivariate_normal(mean, sigma, ngens)[0]
    return s


def dmvnorm(obs, ngens=1):
    x = obs['HO_hypo']
    mean = [obs['H'], obs['O']]
    sigma = [[obs['Hsd'] ** 2, obs['HOc'] * obs['Hsd'] * obs['Osd']],
             [obs['HOc'] * obs['Hsd'] * obs['Osd'], obs['Osd'] ** 2]]
    rv = multivariate_normal(mean, sigma, ngens)
    s = rv.pdf(x)
    return s


def rmvnorm(obs, ngens=10000):
    mean = [obs['H'][0], obs['O'][0]]
    sigma = [[obs['Hsd'][0] ** 2, obs['HOc'][0] * obs['Hsd'][0] * obs['Osd'][0]],
             [obs['HOc'][0] * obs['Hsd'][0] * obs['Osd'][0], obs['Osd'][0] ** 2]]

    s = np.random.multivariate_normal(mean, sigma, ngens)
    return s


def mskfunc(x):
    if x['H_h'] > x['H_obs'] or x['O_h'] > x['O_obs'] or x['S'] <= 0 or x['S'] > 15:
        return 0
    else:
        return 1


def sourceprob(obs, hsource, hslope, ngens=1000, printiter=False):
    """takes values of observed and hypothesized source water (each type 'iso'), hypothesized EL slope value
    prior probability of source, and number of parameter draws

    Args:
        obs: observed source water
        hsource: hypothesized source water
        hslope: hypothesized EL slope value (value, sd)
        ngens: number of parameter draws
        printiter: optional output of iteration prints; default is False

    Returns:
        Pandas Dataframe with index length ngens with new columns:
            hypo_prob: source water prior probability of each draw [P(A)]
            Sprob: relative conditional probability of each draw [P(B|A)]
            slope: slope of each draw
            b: intercept of each draw
    """
    # ngens observed values
    mean = [obs['H'][0], obs['O'][0]]
    sigma = [[obs['Hsd'][0] ** 2, obs['HOc'][0] * obs['Hsd'][0] * obs['Osd'][0]],
             [obs['HOc'][0] * obs['Hsd'][0] * obs['Osd'][0], obs['Osd'][0] ** 2]]

    s = np.random.multivariate_normal(mean, sigma, ngens)
    HO_obs = pd.DataFrame(s, columns=['H_obs', 'O_obs'])

    HO_obs['obs_prob'] = HO_obs[['H_obs', 'O_obs']].apply(lambda x: multivariate_normal.pdf(x, mean=mean, cov=sigma),1)

    # ngens hypothesized source values
    mean = [hsource['H'][0], hsource['O'][0]]
    sigma = [[hsource['Hsd'][0] ** 2, hsource['HOc'][0] * hsource['Hsd'][0] * hsource['Osd'][0]],
             [hsource['HOc'][0] * hsource['Hsd'][0] * hsource['Osd'][0], hsource['Osd'][0] ** 2]]

    s = np.random.multivariate_normal(mean, sigma, ngens)
    HO_h = pd.DataFrame(s, columns=['H_h', 'O_h'])
    HO_h['hypo_prob'] = HO_h[['H_h', 'O_h']].apply(lambda x: multivariate_normal.pdf(x, mean=mean, cov=sigma), 1)

    HO = pd.concat([HO_obs, HO_h], axis=1)
    HO['S'] = (HO['H_obs'] - HO['H_h']) / (HO['O_obs'] - HO['O_h'])

    HO['Sprob'] = norm.pdf(HO['S'], hslope[0], hslope[1]) / norm.pdf(hslope[0], hslope[0], hslope[1])

    HO['msk'] = HO.apply(lambda x: mskfunc(x), 1)
    HO['Sprob'] = HO['msk'] * HO['Sprob']

    goods = HO['msk'].sum()
    HO['b'] = HO['H_obs'] - HO['S'] * HO['O_obs']

    if printiter:
        print(f"{goods} out of {ngens}")
    HO = HO.drop(['msk'], axis=1)
    return HO


#####
# --- MWL source implementation ----
#####
def mwlsource(obs, hslope, mwl=[8.01, 9.57, 167217291.1, 2564532.2, -8.096, 80672], ngens=10000):
    """takes values of observed water (type 'iso'), MWL (see below), hypothesized EL slope value
    and number of parameter draws

    Args:
        obs: observed source water
        mwl: meteoric water line = slope, intercept, sum of squares in d2H, sum of squares in d18O, average d18O, and number of samples.
        hslope: hypothesized EL slope
        ngens: number of parameter draws

    Returns:

    """

    o_cent = (mwl[1] - (obs['H'][0] - hslope[0] * obs['O'][0])) / (hslope[0] - mwl[0])
    o_min = o_cent - 10
    o_max = o_cent + 5
    sr = np.sqrt((mwl[2] - (mwl[0] ** 2 * mwl[3])) / (mwl[5] - 2))

    mean = [obs['H'][0], obs['O'][0]]
    sigma = [[obs['Hsd'][0] ** 2, obs['HOc'][0] * obs['Hsd'][0] * obs['Osd'][0]],
             [obs['HOc'][0] * obs['Hsd'][0] * obs['Osd'][0], obs['Osd'][0] ** 2]]

    HO_dict = {}
    i = 1
    while i <= ngens:
        HO_obs = np.random.multivariate_normal(mean, sigma, 1)[0]
        O_h = o_min + np.random.rand(1) * (o_max - o_min)

        sy = sr * np.sqrt(1 + 1 / mwl[5] + (O_h - mwl[4]) ** 2 / mwl[3])
        H_h = np.random.normal(O_h * mwl[0], sy, 1)
        S = (HO_obs[0] - H_h) / (HO_obs[1] - O_h)
        Sprob = norm.pdf(S, hslope[0], hslope[1]) / norm.pdf(hslope[0], hslope[0], hslope[1])

        if H_h > HO_obs[0] or O_h > HO_obs[1] or S <= 0 or S > 10:
            Sprob = 0
        else:
            pass

        if np.random.rand(1) < Sprob:
            hypo_prob = norm.pdf(H_h, O_h * mwl[0] + mwl[1], sy)
            obs_prob = multivariate_normal.pdf([HO_obs[0], HO_obs[1]], mean, cov=sigma)
            HO_dict[i] = [H_h[0], O_h[0], hypo_prob[0], HO_obs[0], HO_obs[1], obs_prob, Sprob[0]]
            i += 1

    HO = pd.DataFrame.from_dict(HO_dict,
                                orient='index',
                                columns=['H_h', 'O_h', 'hypo_prob', 'H_obs', 'O_obs', 'obs_prob', 'Sprob'])
    return HO


#####
# --- Mixtures implementation ----
#####

# takes values of observed and hypothesized endmember source waters (each type 'iso'),hypothesized EL slope,
# prior (as relative contribution of each source to mixture), and number of parameter draws

def mixprob(obs, hsource, hslope, prior=None, shp=2, ngens=10000):
    """takes values of observed and hypothesized endmember source waters (each type 'iso'),hypothesized EL slope,
    prior (as relative contribution of each source to mixture), and number of parameter draws

    Args:
        obs: observed endmember source water
        hsource: hypothesized endmember source water
        hslope: hypothesized EL slope (slope, slope sd)
        prior: prior (as relative contribution of each source to mixture)
        shp: 2
        ngens: number of parameter draws

    Returns:

    """
    nsource = len(hsource)

    mean = [obs['H'][0], obs['O'][0]]
    sigma = np.array([[obs['Hsd'][0] ** 2, obs['HOc'][0] * obs['Hsd'][0] * obs['Osd'][0]],
             [obs['HOc'][0] * obs['Hsd'][0] * obs['Osd'][0], obs['Osd'][0] ** 2]])
    min_eig = np.min(np.real(np.linalg.eigvals(sigma)))
    if min_eig < 0:
        sigma -= 10*min_eig * np.eye(*sigma.shape)



    if prior is None:
        prior = np.repeat([1], nsource)
    it = 1
    i = 1
    HO_hypo = {}
    while i <= ngens:
        HO_obs = np.random.multivariate_normal(mean, sigma, 1)[0]
        hsource['HO_hypo'] = hsource.apply(lambda x: rmvnormapp(x, 1), 1)
        alphas = prior / np.min(prior) * shp
        fracs = dirichlet.rvs(alphas, size=1)[0]
        H_h = hsource['HO_hypo'].apply(lambda x: x[0] * fracs[0], 1).sum()
        O_h = hsource['HO_hypo'].apply(lambda x: x[1] * fracs[0], 1).sum()
        S = (HO_obs[0] - H_h) / (HO_obs[1] - O_h)
        if (H_h > HO_obs[0]) or (O_h > HO_obs[1]) or S <= 0 or S > 10:
            Sprob = 0
        else:
            Sprob = norm.pdf(S, hslope[0], hslope[1]) / norm.pdf(hslope[0], hslope[0], hslope[1])
        if np.random.rand(1) < Sprob:
            obs_prob = multivariate_normal.pdf(HO_obs, mean, cov=sigma)
            fracs_prob = dirichlet.pdf(fracs, alphas)
            hsource['prob_hold'] = hsource.apply(lambda x: dmvnorm(x), 1)
            hypo_prob = hsource['prob_hold'].product()
            HO_hypo[i] = [H_h, O_h, hypo_prob,
                          HO_obs[0], HO_obs[1],
                          obs_prob, fracs, fracs_prob, Sprob]
            i += 1

        it += 1
        if it > 10000 and i / it < 0.01:
            print("too few valid draws")
            break
    HO = pd.DataFrame.from_dict(HO_hypo,
                                orient='index',
                                columns=['H_h', 'O_h', 'hypo_prob',
                                         'H_obs', 'O_obs',
                                         'obs_prob', 'fracs', 'fracs_prob', 'Sprob'])
    print(f"{it} iterations for {ngens} posterior samples")
    return HO
