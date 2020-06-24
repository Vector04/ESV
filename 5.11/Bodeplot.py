import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import matplotlib.ticker as ticker
from scipy import stats

import floats
from helpers import *


R, C = 1e4, 1e-6


# Parsing our data
names = ['freq', 'mag', 'phase']
df = pd.read_csv('Bode plot 1.csv', names=names, header=0)


# Going all out, using a double y axis
fig, ax = plt.subplots()
ax2 = ax.twinx()

freqs = np.logspace(0, 10, 200, base=10)
As = 1 / (R * C * freqs * 2 * np.pi)

# Magnitude
ax.plot(df['freq'], 20 * np.log10(df['mag']),
        label='Data (Amplitude)', linewidth=2, color='#1e0acf')
ln = ax.plot(freqs, 20 * np.log10(As), '--',
             label='Voorspelling (Amplitude)', linewidth=1, color='#2986ff')
add_arrow(ln[0], position=9e9)

# Phase
ax2.plot(df['freq'], df['phase'] / 180,
         label='Data (fase)', linewidth=2, color='#00940f')
ln = ax2.plot(freqs, np.ones(200) / 2, '--',
              label='Voorspelling (fase)', linewidth=1, color='#a8ff80')
add_arrow(ln[0], position=8e9)


ax.set_xlabel('Frequentie (Hz)', fontsize=12)
ax.set_ylabel('Amplitude (dB)', fontsize=12)
ax2.set_ylabel('Fase (rad)')
ax.set_title('Integrator transfer', fontsize=14)

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
ax.set_yticks(np.arange(-200, 41, 40))
ax2.set_yticks(np.arange(0, 0.61, 0.1))
ax2.set_ylim(0, 0.6)


ticks_y2 = ticker.FuncFormatter(lambda x, pos: 0 if x == 0 else fr'{x:g}$\pi$')
ax2.yaxis.set_major_formatter(ticks_y2)

ax.tick_params(color='#2811f5', axis='y')
ax2.tick_params(color='#22992e', axis='y')

ax2.spines['left'].set_color('#2811f5')
ax2.spines['right'].set_color('#22992e')

# Grid
ax.grid(which='major', alpha=0.6)
ax.grid(which='minor', alpha=0.2)

plt.savefig('5.png')
plt.show()
