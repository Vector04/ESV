import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import matplotlib.ticker as ticker
from scipy import stats

import inspect
import floats
from helpers import *


R, C = 1e4, 1e-6

# First, a modified version of curve_fit to save some work:


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


# Parsing our data
names = ['freq', 'mag', 'phase']
df = pd.read_csv('Bode plot 1.csv', names=names, header=0)


fig, ax = plt.subplots()

freqs = np.logspace(0, 10, 100, base=10)
As = 1 / (R * C * freqs * 2 * np.pi)
print(As)

ax.plot(df['freq'], 20 * np.log10(df['mag']), label='Data')
ax.plot(freqs, 20 * np.log10(As), label='Voorspelling')


ax.set_xlabel('Frequentie (Hz)', fontsize=12)
ax.set_ylabel('Amplitude (dB)', fontsize=12)
ax.set_title('Integrator Amplitude transfer', fontsize=14)
plt.legend()


ax.set_xscale('log')
major_ticks = [10**x for x in range(11)]
minor_ticks = [10**x * y for x in range(10) for y in range(1, 10)]
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticklabels(['1', '10', '100', '1k', '10k',
                    '100k', '1M', '10M', '100M', '1G', '10G'])

ax.set_xlim(1, 1e10)
ax.set_ylim(-100, 40)
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

# Grid
ax.grid(which='major', alpha=0.69)
ax.grid(which='minor', alpha=0.2)

plt.savefig("5.png")
plt.show()

# # Big thank you to https://stackoverflow.com/users/7296115/floriaan for the following function:


# def calculate_ticks(ax, ticks, round_to=0.1, center=False):
#     upperbound = np.ceil(ax.get_ybound()[1] / round_to)
#     lowerbound = np.floor(ax.get_ybound()[0] / round_to)
#     dy = upperbound - lowerbound
#     fit = np.floor(dy / (ticks - 1)) + 1
#     dy_new = (ticks - 1) * fit
#     if center:
#         offset = np.floor((dy_new - dy) / 2)
#         lowerbound = lowerbound - offset
#     values = np.linspace(lowerbound, lowerbound + dy_new, ticks)
#     return values * round_to


# # Setting our 2 tick-matching scales was never easier:
# ax.set_yticks(calculate_ticks(ax, 9, center=True, round_to=0.05))

# ax.grid(which='major', alpha=0.6)

# plt.savefig('10.png')
# plt.show()
