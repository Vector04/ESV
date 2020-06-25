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
df500 = pd.read_csv('squarewaves/500.csv',
                    usecols=[0,1, 4], header=0, names=names)
df1k = pd.read_csv('squarewaves/1k.csv',
                   usecols=[0,1, 4], header=0, names=names)
df10k = pd.read_csv('squarewaves/10k.csv',
                    usecols=[0,1, 4], header=0, names=names)


fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True,
                               gridspec_kw={'width_ratios': [2, 1]})

# plotting
ax1.plot(df500['t'], df500['Vin'], label=r'$V_{in}$')
ax1.plot(df500['t'], df500['Vout'], label=r'$R_C = 500\ \Omega$')
ax1.plot(df1k['t'], df1k['Vout'], label=r'$R_C = 1$ k$\Omega$')
ax1.plot(df10k['t'], df10k['Vout'], label=r'$R_C = 10$ k$\Omega$')

ax2.plot(df500['t'], df500['Vin'], label=r'$V_{in}$')
ax2.plot(df500['t'], df500['Vout'], label=r'$R_C = 500\ \Omega$')
ax2.plot(df1k['t'], df1k['Vout'], label=r'$R_C = 1$ k$\Omega$')
ax2.plot(df10k['t'], df10k['Vout'], label=r'$R_C = 10$ k$\Omega$')

ax1.set_xlim(0, 0.01)
ax2.set_xlim(0.05, 0.055)

ax1.legend()
ax1.set_ylabel('Voltage (V)')
ax1.grid()
ax2.grid()


fig.suptitle('Gedrag integrator blokgolf', fontsize=14)
ax1.set_title(r'Vanaf $t=0$', fontsize=12)
ax2.set_title('In evenwicht', fontsize=12)
ax1.set_xlabel(r'Tijd ($m$s)')
ax2.set_xlabel(r'Tijd ($m$s)')

# time unit of ms
ticks_ax1x = ticker.FuncFormatter(lambda x, pos: fr'{x*1000}')
ax1.xaxis.set_major_formatter(ticks_ax1x)
ticks_ax2x = ticker.FuncFormatter(lambda x, pos: fr'{x*1000:.1f}')
ax2.xaxis.set_major_formatter(ticks_ax2x)


ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.001))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.001))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))


plt.savefig('9.png')
plt.show()

# fig, ax = plt.subplots()


# def f(t, v0, phi, k, tau):
#     return v0 * np.sin(1000 * np.pi * t + phi) + k * np.exp(t / tau)


# def better_curve_fit(f, xdata, ydata, p0=None, sigma=None, **kwargs):
#     """
#     A modified version of curve_fit which returns more useful information.

#     """
#     global popt
#     popt, pcov = optimize.curve_fit(
#         f, xdata, ydata, p0=p0, sigma=sigma, **kwargs)
#     argspec = inspect.getfullargspec(f)
#     params = ParamDict({param: floats.floatE(val, error)
#                         for param, val, error in zip(argspec[0][1:], popt, np.sqrt(np.diag(pcov)))})
#     return params, popt


# params, popt = better_curve_fit(
#     f, df10k['t'], df10k['Vout'], p0=(0.3, np.pi / 2, -0.1, -0.1))
# print(params)


# def f2(t, a, b, k, tau):
#     return k * np.exp(t / tau)


# ax.plot(df10k['t'], df10k['Vout'], c='#33aaff')
# ax.plot(df10k['t'], f2(df10k['t'], *popt), '--', c='#f55656')
# plt.show()
