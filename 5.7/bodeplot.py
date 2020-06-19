import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker


# First, parsing the data
names = ['freq', 'mag', 'phase']
df = pd.read_csv('bodeplot.csv', names=names, header=0)
print(df)


# We can predict the way our bode plot will look by using the complex tranfer function
# R, C, Rc
params = (1e4, 1e-6, 1e4)


def A(w, R, C, Rc):
    return (Rc / R) * (1) / (1 + 1j * w * Rc * C)


ts = np.logspace(0, 10, 100, base=10)
predictions = 20 * np.log10(np.abs(A(ts, *params)))

fig, ax = plt.subplots()

ax.plot(df['freq'], 20 * np.log10(df['mag']), label='Data Multisim')
ax.plot(ts, predictions, label='Voorspelling')


# Time to make our plot look ok
plt.legend()
ax.set_xlabel('Frequentie (Hz)', fontsize=12)
ax.set_ylabel('Amplitude (dB)', fontsize=12)
ax.set_title('Amplitude Transfer Integrator', fontsize=14)

# # Proper Scales
ax.set_xscale('log')
major_ticks = [10**x for x in range(11)]
minor_ticks = [10**x * y for x in range(10) for y in range(1, 10)]
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticklabels(['1', '10', '100', '1k', '10k',
                    '100k', '1M', '10M', '100M', '1G', '10G'])
ax.set_xlim(1, 1e10)
ax.set_ylim(-120, 10)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

ax.grid(which='major', alpha=0.69)
ax.grid(which='minor', alpha=0.2)

plt.savefig('11.png')

# plt.show()


# Now, for the fase transfer:
fig, ax = plt.subplots()

predictions = np.angle(A(ts, *params)) + np.pi
ax.plot(df['freq'], df['phase'] * np.pi / 180, label='Data Multisim')
ax.plot(ts, predictions, label='Voorspelling')

# Make our plot look decent
plt.legend()
ax.set_xlabel('Frequentie (Hz)', fontsize=12)
ax.set_ylabel('Fase (rad)', fontsize=12)
ax.set_title('Fase transfer Integrator', fontsize=14)

# Proper logscale on the x-axis
ax.set_xscale('log')
major_ticks = [10**x for x in range(11)]
minor_ticks = [10**x * y for x in range(10) for y in range(1, 10)]
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticklabels(['1', '10', '100', '1k', '10k',
                    '100k', '1M', '10M', '100M', '1G', '10G'])
ax.set_xlim(1, 1e10)

# Proper y scale in radians
ticks_y = ticker.FuncFormatter(lambda x, pos: r'{0:g}$\pi$'.format(x / np.pi))
ax.yaxis.set_major_formatter(ticks_y)
ax.set_yticks(np.pi * np.arange(0, 1.2, 0.1))

ax.set_ylim(0, 1.1 * np.pi)

ax.grid(which='major', alpha=0.69)
ax.grid(which='minor', alpha=0.2)

plt.savefig('12.png')

plt.show()
