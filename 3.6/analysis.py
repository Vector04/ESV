import numpy as np
from scipy import optimize
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


df = pd.read_csv('data.csv', names=['Rs', 'Vs', 'dVs'])

df['Vs'] /= 1000  # Proper unit
df['dVs'] /= 1000  # Proper units

# Let's do a fit of the data


def f(R, Ri, V0):
    return R / (R + Ri) * V0


params, chi2 = better_curve_fit(
    f, df['Rs'], df['Vs'], p0=(1000, 1), sigma=df['dVs'])

print(params)
print(chi2)


xs = np.linspace(500, 50000, 200)
plt.plot(xs, f(xs, *popt), label='fit')
plt.errorbar(df['Rs'], df['Vs'], yerr=10 * df['dVs'],
             fmt='o', markersize=2.5, label='data')
plt.xlabel('Weerstand (Ω)')
plt.ylabel('Voltage (V)')
plt.title('Voltage verloop')
plt.grid()

plt.savefig('11.png')
plt.show()
