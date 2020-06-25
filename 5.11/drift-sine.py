import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import matplotlib.ticker as ticker
from scipy import stats

import floats
from helpers import *

# first, parsing our data
names = ['t', 'Vout', 'Vin']
df500 = pd.read_csv('sinewaves/500.csv',
                    usecols=[0,1, 4], header=0, names=names, nrows=600)
df1k = pd.read_csv('sinewaves/1k.csv',
                   usecols=[0,1, 4], header=0, names=names, nrows=600)
df10k = pd.read_csv('sinewaves/10k.csv',
                    usecols=[0,1, 4], header=0, names=names, nrows=600)


fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, sharey=True)

# plotting
ax1.plot(df500['t'], df500['Vout'], label=r'$R_C = 500\ \Omega$', c='#33aaff')
ax2.plot(df1k['t'], df1k['Vout'], label=r'$R_C = 1$ k$\Omega$', c='#33aaff')
ax3.plot(df10k['t'], df10k['Vout'], label=r'$R_C = 10$ k$\Omega$', c='#33aaff')

# General formatting
for ax in (ax1, ax2, ax3):
    ax.set_xlim(0, 0.035)
    ax.set_ylim(-0.65, 0.65)

    ax.legend(loc=1)
    ax.set_ylabel(r'$V_{out}$ (V)')
    ax.grid()


ax1.set_title('Gedrag integrator sinusgolf', fontsize=14)
ax3.set_xlabel(r'Tijd ($m$s)')
# time unit of ms
ticks_ax3 = ticker.FuncFormatter(lambda x, pos: fr'{x*1000}')
ax3.xaxis.set_major_formatter(ticks_ax3)


plt.savefig('8.png')
# plt.show()

fig, ax = plt.subplots()


def f(t, v0, phi, k, tau):
    return v0 * np.sin(1000 * np.pi * t + phi) + k * np.exp(t / tau)


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


params, popt = better_curve_fit(
    f, df10k['t'], df10k['Vout'], p0=(0.3, np.pi / 2, -0.1, -0.1))
print(params)


def f2(t, a, b, k, tau):
    return k * np.exp(t / tau)


ax.plot(df10k['t'], df10k['Vout'], c='#33aaff')
ax.plot(df10k['t'], f2(df10k['t'], *popt), '--', c='#f55656')
plt.show()
