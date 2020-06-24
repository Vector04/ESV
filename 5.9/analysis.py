from fractions import Fraction as Fr

import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# A misc. function to add an arrow to a line, for convenience
# https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib


def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[end_ind], ydata[end_ind]),
                       arrowprops=dict(arrowstyle="->", color=color),
                       size=size
                       )


# A few basic constants for predicting A(w)
R, C = 1e3, 1e-6

# Parsing our data
names = ['freq', 'mag', 'phase']
df = pd.read_csv('Bode plot data.csv', names=names, header=0)
print(df)


def A_abs(f):
    """Predicted amplitude of differentiator"""
    w = 2 * np.pi * f
    return R * C * w


freqs = np.logspace(0, 10, 100, base=10)
predictions = 20 * np.log10(A_abs(freqs))

# Plotting our data
fig, ax = plt.subplots()
ax.plot(df['freq'], 20 * np.log10(df['mag']), label='Data', linewidth=2)
ln = ax.plot(freqs, predictions, '--', label='Voorspelling', linewidth=1)
add_arrow(ln[0], position=1.1e8)

# Make our graph look ok
ax.set_xlabel('Frequentie (Hz)', fontsize=12)
ax.set_ylabel('Amplitude (dB)', fontsize=12)
ax.set_title('Differentiator Amplitude Transfer', fontsize=14)
plt.legend()

# Proper axis formatting
ax.set_xscale('log')
major_ticks = [10**x for x in range(11)]
minor_ticks = [10**x * y for x in range(10) for y in range(1, 10)]
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticklabels(['1', '10', '100', '1k', '10k',
                    '100k', '1M', '10M', '100M', '1G', '10G'])

ax.set_xlim(1, 1e10)
ax.set_ylim(-40, 120)
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

# Grid
ax.grid(which='major', alpha=0.69)
ax.grid(which='minor', alpha=0.2)

plt.savefig("3.png")


# Now, we also want to take a look at the phase of A(w):
# Creating a seperator plot
fig, ax = plt.subplots()

# Plotting our data
ax.plot(df['freq'], df['phase'] / 180 * np.pi, label='data', linewidth=2)
ln = ax.plot(freqs, [-np.pi / 2] * 100, '--',
             label='Voorspelling', linewidth=1)
add_arrow(ln[0], position=8e9)

# Basic elements
ax.set_xlabel('Frequentie (Hz)', fontsize=12)
ax.set_ylabel('Fase (rad)', fontsize=12)
ax.set_title('Differentiator Fase Transfer', fontsize=14)
plt.legend()

# Axis formatting
ax.set_xscale('log')
major_ticks = [10**x for x in range(11)]
minor_ticks = [10**x * y for x in range(10) for y in range(1, 10)]
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticklabels(['1', '10', '100', '1k', '10k',
                    '100k', '1M', '10M', '100M', '1G', '10G'])
ax.set_xlim(1, 1e10)

# We'd like our y axis formatted properly, the y axis (in radians) is formatted
# as fractional multiples of pi, for this we use the fractions.Fractions class.
# We do then need to make an exception for y = 0.
ticks_y = ticker.FuncFormatter(
    lambda x, pos: 0 if pos == 8 else str(Fr(x / np.pi)) + r"$\pi$")
ax.yaxis.set_major_formatter(ticks_y)
ax.set_yticks(np.pi * np.arange(-2, 0.1, 0.25))
ax.set_ylim(top=0)

# Grid
ax.grid(which='major', alpha=0.69)
ax.grid(which='minor', alpha=0.2)

plt.savefig("4.png")
# plt.show()

# As a last note, I'd like to know the actual slope of A(w):


def f(f, a):
    return 2 * np.pi * a * f


df2 = df.iloc[:44]
print(df2)
popt, pcov = optimize.curve_fit(f, df2['freq'], df2['mag'])
print(popt)
print()
print(np.sqrt(np.diag(pcov)))

print(np.mean(df2['phase'] / 180) , np.std(df2['phase'] / 180))
