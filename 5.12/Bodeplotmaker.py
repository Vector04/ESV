import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import matplotlib.ticker as ticker
from scipy import stats
from fractions import Fraction

from helpers import *

# R, C_A, C_B
params = (1 / (np.sqrt(2) * 1e-6 * 2000 * np.pi), 2e-6, 1e-6)
print(params)


# Parsing our data
names = ['freq', 'mag', 'phase']
df = pd.read_csv('bodeplotdata.csv', names=names, header=0)

# Making some predictions:


def A(f, R, C_A, C_B):
  w = 2 * np.pi * f
  return (1 + (2j * w * R * C_B) - (w**2 * R**2 * C_A * C_B))**(-1)


freqs = np.logspace(0, 10, 200, base=10)
As = A(freqs, *params)


# Going all out, using a double y axis
fig, ax = plt.subplots()
ax2 = ax.twinx()
fig.set_size_inches(1.1 * fig.get_size_inches())

# Magnitude
ax.plot(df['freq'], 20 * np.log10(df['mag']),
        label='Data (Amplitude)', linewidth=2, color='#1e0acf')
ln = ax.plot(freqs, 20 * np.log10(np.abs(As)), '--',
             label='Voorspelling (Amplitude)', linewidth=1, color='#2986ff')
add_arrow(ln[0], position=9e5)

# Phase
ax2.plot(df['freq'], df['phase'] / 180,
         label='Data (fase)', linewidth=2, color='#00940f')
ln = ax2.plot(freqs, np.angle(As) / np.pi, '--',
              label='Voorspelling (fase)', linewidth=1, color='#a8ff80')
add_arrow(ln[0], position=9e9)


ax.set_xlabel('Frequentie (Hz)', fontsize=12)
ax.set_ylabel('Amplitude (dB)', fontsize=12)
ax2.set_ylabel('Fase (rad)', fontsize=12)
ax.set_title('Sallen & Key transfer', fontsize=14)

lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc=3)

# x axis formatting
ax.set_xscale('log')
major_ticks = [10**x for x in range(11)]
minor_ticks = [10**x * y for x in range(10) for y in range(1, 10)]
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticklabels(['1', '10', '100', '1k', '10k',
                    '100k', '1M', '10M', '100M', '1G', '10G'])

ax.set_xlim(1, 1e10)

# Manual y-tick formatting for aligned grid
ax.set_yticks(np.arange(-120, 21, 20))
ax.set_ylim(-120, 20)
ax2.set_yticks(np.arange(-2, 0.4, 1 / 3))
ax2.set_ylim(-2, 1 / 3)


def smart_formatter(x, pos):
  if abs(x) < 0.01:
    return '0'
  if abs(x + 1) < 0.01:
    return r'$-\pi$'
  if x == -2:
    return r'$-2\pi$'
  return fr"{int(round(3 * x))}/3$\pi$"


ticks_y2 = ticker.FuncFormatter(smart_formatter)
ax2.yaxis.set_major_formatter(ticks_y2)

ax.tick_params(color='#2811f5', axis='y')
ax2.tick_params(color='#22992e', axis='y')

ax2.spines['left'].set_color('#2811f5')
ax2.spines['right'].set_color('#22992e')

# Grid
ax.grid(which='major', alpha=0.6)
ax.grid(which='minor', alpha=0.2)

plt.savefig('11.png')
plt.show()
