import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import matplotlib.ticker as ticker
from scipy import stats

import inspect
import floats
from helpers import *


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
names = ['t', 'Vout', 'Vin']
df = pd.read_csv('Time series.csv', usecols=[0,1,4], names=names, header=0)

# Selecting the rows with equilibrium
df = df.iloc[1290:]
print(df)

# We want to do a fit of both time series, and check the phase and amplitude difference. If those are as predicted, we can confirm our derived equations were correct.


def f(t, A, w, phi):
    return A * np.sin(w * t + phi)


params_in, popt_in = better_curve_fit(
    f, df['t'], df['Vin'], p0=(1, 2000 * np.pi, 0))
print(params_in)
A_in, phi_in = params_in['A'], params_in['phi']

params_out, popt_out = better_curve_fit(
    f, df['t'], df['Vout'], p0=(0.02, 2000 * np.pi, np.pi / 2))
print(params_out)
A_out, phi_out = params_out['A'], params_out['phi']

# The difference in phase:
dphi = (phi_out - phi_in) / np.pi
print(f"A phase differnce of {dphi} π")
print()
print("Amplitude ratio:")
print((A_out / A_in))
print("predicted amplitude ratio:")
print(1 / (2000 * np.pi * 1e4 * 1e-6))

print(((A_out / A_in) / (1 / (2000 * np.pi * 1e4 * 1e-6)) - 1) * 100)


w_in, w_out = params_in['w'], params_out['w']
print(w_in / (2 * np.pi))
print(w_out / (2 * np.pi))

# Now, everybody's favorite part, plotting.
# Note, this was actually a bit more difficult due to scaling issues: A_in \approx 50 A_out.
# I tried to keep the scaling as simple as possible.
# I resorted to use a scondary y-axis for V_out on the left.
fig, ax = plt.subplots()
ax2 = ax.twinx()

df_plot = df.iloc[:100]
ts = np.linspace(0.04, 0.043, 100)

ax.scatter(df_plot['t'], df_plot['Vin'],
           label=r'$V_{in}$', c='green', s=5)
ax.plot(ts, f(ts, *popt_in), label=r'fit of $V_{in}$', c='darkgreen')
ax2.scatter(df_plot['t'], df_plot['Vout'],
            label=r'$V_{out}$', c='orange', s=5)
ax2.plot(ts, f(ts, *popt_out),
         label=r'fit of $V_{out}$', c='darkorange')


ax.set_xlabel(r'Tijd ($ms$)', fontsize=12)
ax.set_ylabel(r'Spanning $V_{in}$ (V)', fontsize=12)
ax2.set_ylabel(r'Spanning $V_{out}$ ($m$V)', fontsize=12)
ax.set_title('Integrator signaal sample', fontsize=14)
ax.set_xlim(0.04, 0.043)

# All labels need to be in one legend, this takes some more code:
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc=1)

# Using ms, mV for better scaling (I hope)
ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 1000))
ax.xaxis.set_major_formatter(ticks_x)
ax2.yaxis.set_major_formatter(ticks_x)

# Proper tick formatting for the x-axis
major_ticks = np.arange(0.04, 0.043, 0.0004)
minor_ticks = np.arange(0.04, 0.043, 0.0001)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)

ticker.LinearLocator(numticks=9)

# Big thank you to https://stackoverflow.com/users/7296115/floriaan for the following function:


def calculate_ticks(ax, ticks, round_to=0.1, center=False):
    upperbound = np.ceil(ax.get_ybound()[1] / round_to)
    lowerbound = np.floor(ax.get_ybound()[0] / round_to)
    dy = upperbound - lowerbound
    fit = np.floor(dy / (ticks - 1)) + 1
    dy_new = (ticks - 1) * fit
    if center:
        offset = np.floor((dy_new - dy) / 2)
        lowerbound = lowerbound - offset
    values = np.linspace(lowerbound, lowerbound + dy_new, ticks)
    return values * round_to


# Setting our 2 tick-matching scales was never easier:
ax.set_yticks(calculate_ticks(ax, 9, center=True, round_to=0.05))
ax2.set_yticks(calculate_ticks(ax2, 9, center=True, round_to=0.005))

ax.grid(which='major', alpha=0.6)

plt.savefig('10.png')
plt.show()
