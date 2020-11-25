"""
Scripts based on Estimation of evaporative loss based on the stable isotope composition
of water using Hydrocalculator by Skrzypek
"""


from scipy import optimize
import pandas as pd
import numpy as np

class CraigGordonModel(object):
    def __init__(self, df, t='tmean (degrees C)',
                 h='hum', l='dl', p='dp', rain='d', slp=None):
        self.t = t  # temperature
        self.h = h  # humidity

        self.lel = slp  # LEL
        self.df = df.copy(deep=True)
        self.isos = ['2H', '18O']
        self.df['X'] = 1.
        self.X = 1.

        if f'{rain}{self.isos[0]}' in self.df.columns and f'{rain}{self.isos[1]}':
            self.rain = rain
        else:
            self.rain = None

        if f'{l}{self.isos[0]}' in self.df.columns and f'{l}{self.isos[1]}':
            self.l = l
        else:
            self.l = None

        if f'{p}{self.isos[0]}' in self.df.columns and f'{p}{self.isos[1]}':
            self.p = p
        else:
            self.p = None

        self.alph_eps()

    def get_f(self):
        """

        Returns:
            Result for non-steady-state model: evaporated fraction of the volume
        """
        self.est_lel()
        for iso in self.isos:
            self.df[f'f (d{iso})'] = self.est_nss(iso)

    def est_nss(self, iso):
        """Non-steady State Model to estimate evaporative loss fraction of the pool volume
        Args:
            iso: isotope species of interest; ex. '2H' or '18O'; default is '2H'

        Returns:
            Result for non-steady-state model: evaporated fraction of the volume
        """
        return (1 - ((self.df[f'{self.l}{iso}'] - self.delta_star(iso)) / (
                    self.df[f'{self.p}{iso}'] - self.delta_star(iso))) ** (1 / self.est_m(iso)))

    def est_lel(self):
        if self.lel:
            self.df['X'] = self.runopt()
        self.df['LEL'] = self.df.apply(lambda x: self.est_slope_alt(x, x['X']), 1)

    def est_ei(self):
        self.est_lel()
        for iso in self.isos:
            self.df[f'm{iso}'] = self.est_m(iso)
            self.df[f'E/I{iso}'] = self.est_e_ovr_i(iso)

    def alph_eps(self):
        # convert temperature from Celcius to Kelvin
        self.df['T (K)'] = self.df[self.t] + 273.15

        # from Horita and Wesolowski (1994)
        self.df['1000 lnα+ (2H)'] = 1158.8 * (self.df['T (K)'] ** 3 / 1000000000) - 1620.1 * (
                    self.df['T (K)'] ** 2 / 1000000) + 794.84 * (self.df['T (K)'] / 1000) - 161.04 + 2.9992 * (
                                                1000000000 / self.df['T (K)'] ** 3)
        self.df['1000 lnα+ (18O)'] = -7.685 + 6.7123 * (1000 / self.df['T (K)']) - 1.6664 * (
                    1000000 / self.df['T (K)'] ** 2) + 0.35041 * (1000000000 / self.df['T (K)'] ** 3)

        # calculate equilibrium isotope fractionation factors (temperature dependent)
        for iso in self.isos:
            self.df[f'α+ ({iso})'] = self.alpha_plus(iso)
            self.df[f'ε+ ({iso})'] = self.epsilon_plus(iso)

        # kinetic isotope fractionation factors (‰) (humidity dependent)
        self.df['εk (δ2H ‰)'] = 12.5 * (1 - self.df[self.h])
        self.df['εk (δ18O ‰)'] = 14.2 * (1 - self.df[self.h])

        for iso in self.isos:
            # Total isotope fractionation (‰)
            self.df[f'ε (δ{iso} ‰)'] = self.est_epsilon(iso=iso)
            # enrichment slope
            self.df[f'm (δ{iso})'] = self.est_m(iso=iso)

    def epsilon_plus(self, iso='2H'):
        """Calculates Equilibrium isotope fractionation factor(‰) (T dependent)

        Args:
            iso: isotope species of interest; ex. '2H' or '18O'; default is '2H'

        Returns:
            Equilibrium isotope fractionation factor(‰) (T dependent)
        """
        return (self.df[f'α+ ({iso})'] - 1) * 1000

    def alpha_plus(self, iso='2H'):
        """Equilibrium isotope fractionation factor (‰) (T dependent)

        Args:
            iso: isotope species of interest; ex. '2H' or '18O'; default is '2H'

        Returns:
            Equilibrium isotope fractionation factor (‰) (T dependent)
        """
        return (np.exp(self.df[f'1000 lnα+ ({iso})'] / 1000))

    def est_epsilon(self, iso='2H'):
        """Total isotope fractionation (‰)

        Args:
            iso: isotope species of interest; ex. '2H' or '18O'; default is '2H'

        Returns:
            Total isotope fractionation (‰)
        """
        return self.df[f'ε+ ({iso})'] / self.df[f'α+ ({iso})'] + self.df[f'εk (δ{iso} ‰)']

    def est_m(self, iso='2H'):
        return (self.df[self.h] - self.df[f'ε (δ{iso} ‰)'] / 1000) / (
                    1 - self.df[self.h] + (self.df[f'εk (δ{iso} ‰)'] / 1000))

    def est_e_ovr_i(self, iso='2H'):
        """

        Args:
            iso: isotope species of interest; ex. '2H' or '18O'; default is '2H'

        Returns:
            Result for steady-state model: evaporation over Inflow ratio
        """
        return (self.df[f'{self.l}{iso}'] - self.df[f'{self.p}{iso}']) / (
                    (self.delta_star(iso) - self.df[f'{self.l}{iso}']) * self.est_m(iso))

    def delta_star(self, iso='2H'):
        return (self.df[self.h] * [self.dAalt(self.df.loc[i], iso=iso, X=self.df.loc[i, 'X']) for i in self.df.index] +
                self.df[f'ε (δ{iso} ‰)']) / (self.df[self.h] - (self.df[f'ε (δ{iso} ‰)'] / 1000))

    def dAalt(self, data, iso='2H', X=1.):
        return (data[f'd{iso}'] - X * data[f'ε+ ({iso})']) / (1 + X * data[f'ε+ ({iso})'] / 1000)

    def est_slope_alt(self, data, X=1.):
        """estimate of the LEL

        Args:
            data: dataframe with relevant isotope data
            X: adjustment factor; defaults to 1.

        Returns:
            LEL
        """
        return self.slope_part_alt(data, iso='2H', X=X) / self.slope_part_alt(data, iso='18O', X=X)

    def slope_part_alt(self, data, iso='2H', X=1.):
        return (data[self.h] * (self.dAalt(data, iso, X) / 1000 - data[f'd{iso}'] / 1000) + (
                    1 + data[f'd{iso}'] / 1000) / 1000 * data[f'ε (δ{iso} ‰)']) / (
                           data[self.h] - data[f'ε (δ{iso} ‰)'] / 1000)

    def minslope(self, X, data):
        """Input function for optimization of X"""
        return self.lel - self.est_slope_alt(data, X=X)

    def optx(self, data):
        """Optimization function that minimizes difference between input LEL and estimated LEL by adjusting X, which modifies epsilon plus

        Args:
            data: dataframe with relevant isotope data

        Returns:
            optimized value of X
        """
        res = optimize.minimize(self.minslope, args=(data), x0=[self.X], bounds=[(0.6, 1.0)],
                                options={'maxiter': 10000})
        return res.x[0]

    def runopt(self):
        return self.df.apply(lambda x: self.optx(x), 1)


