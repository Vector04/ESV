import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import matplotlib.ticker as ticker
from scipy import stats

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
    return params, popt


pd.options.display.max_rows = 999
names = ['freq', 'mag', 'phase']
df = pd.read_csv('Bode plot 1.csv', names=names, header=0)

print(df)

R, C = 1e4, 1e-6

# We want to curve fit to |A(w)| to get the integration constant
df2 = df.iloc[:50]


def f(f, RC):
    w = 2 * np.pi * f
    return 1 / (RC * w)


params, popt = better_curve_fit(f, df['freq'], df['mag'], p0=(1e-2))
print(params)
