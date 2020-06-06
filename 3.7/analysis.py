import numpy as np
from scipy import optimize
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
import inspect

import floats
from helpers import *


def better_curve_fit(f, xdata, ydata, p0=None, sigma=None, **kwargs):
    """
    A modified version of curve_fit which returns more useful information.

    """
    global popt
    popt, pcov = optimize.curve_fit(
        f, xdata, ydata, p0=p0, sigma=sigma, **kwargs)
    argspec = inspect.getfullargspec(f)
    params = ParamDict({param: floats.floatE(val, error)
                        for param, val, error in zip(argspec[0][1:], popt, np.sqrt(np.diag(pcov)))})

    chi2 = chi_squared(lambda x: f(x, *popt), xdata, ydata,
                       sigma) / (len(xdata - len(popt)))
    return params, chi2


df = pd.read_csv('data.csv')

plt.scatter(df['dR'], df['Vab'], c='orange', zorder=100, label='data', s=25)


def L(dR, R, Vext=15):
    return ((R + dR) / (2 * R + dR) - (1 / 2)) * Vext * 1000


params, chi2 = better_curve_fit(
    L, df['dR'], df['Vab'], p0=(1500), sigma=df['dVab'])
print(params)
print(chi2)
z = stats.norm.ppf(stats.chi2.sf(chi2, len(df['dR']) - 1))
print(z)

dRs = np.linspace(-100, 100, 100)
plt.plot(dRs, L(dRs, 2000), label='Theorie')
plt.xlabel('Delta weerstand (Ω)')
plt.ylabel('Voltage over ab (mV)')
plt.title('Voltageverloop Wheatstone Brug')
plt.legend()
plt.grid()
plt.savefig('14.png')
plt.show()
