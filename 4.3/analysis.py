import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.ticker import MultipleLocator


# Getting the data
names = ['freq', 'mag', 'phase']

df01 = pd.read_csv('alpha0.1.csv', names=names, header=0)
df1 = pd.read_csv('alpha1.csv', names=names, header=0)
df10 = pd.read_csv('alpha10.csv', names=names, header=0)


print(df10)


fig, ax = plt.subplots()
ax.plot(df01['freq'], 20 * np.log10(df01['mag']), label='α = 0.1')
ax.plot(df1['freq'], 20 * np.log10(df1['mag']), label='α = 1')
ax.plot(df10['freq'], 20 * np.log10(df10['mag']), label='α = 10')


plt.legend()
ax.set_xlabel('Frequentie (Hz)')
ax.set_ylabel('Amplitude (dB)')
ax.set_title('Amplitude Transfer')

# Proper Scales
ax.set_xscale('log')
major_ticks = [10**x for x in range(9)]
minor_ticks = [10**x * y for x in range(9) for y in range(1, 10)]
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticklabels(['1', '10', '100', '1k', '10k',
                    '100k', '1M', '10M', '100M'])
ax.set_xlim(1, 1e8)
ax.set_ylim(-100, 0)
ax.yaxis.set_major_locator(MultipleLocator(10))

ax.grid(which='major', alpha=0.6)
ax.grid(which='minor', alpha=0.2)

plt.savefig('7.png')


plt.show()
