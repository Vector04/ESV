import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.ticker import MultipleLocator


# Getting the data
names = ['freq', 'mag', 'phase']

df0 = pd.read_csv('R0.csv', names=names, header=0)
df1 = pd.read_csv('R1.csv', names=names, header=0)
df7 = pd.read_csv('R7.csv', names=names, header=0)
df15 = pd.read_csv('R15.csv', names=names, header=0)


print(df0)


fig, ax = plt.subplots()
ax.plot(df0['freq'], 20 * np.log10(df0['mag']), label='R = 0')
ax.plot(df1['freq'], 20 * np.log10(df1['mag']), label='R = 1')
ax.plot(df7['freq'], 20 * np.log10(df7['mag']), label='R = 7')
ax.plot(df15['freq'], 20 * np.log10(df15['mag']), label='R = 15')


plt.legend()
ax.set_xlabel('Frequentie (Hz)')
ax.set_ylabel('Amplitude (dB)')
ax.set_title('Amplitude Transfer')

# Proper Scales
ax.set_xscale('log')
major_ticks = [10**x for x in range(9)]
minor_ticks = [10**x * y for x in range(7) for y in range(1, 10)]
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticklabels(['1', '10', '100', '1k', '10k',
                    '100k', '1M'])
ax.set_xlim(1, 1e6)
ax.set_ylim(-110, 10)
ax.yaxis.set_major_locator(MultipleLocator(10))

ax.grid(which='major', alpha=0.6)
ax.grid(which='minor', alpha=0.2)

plt.savefig('15_v2.png')


plt.show()
