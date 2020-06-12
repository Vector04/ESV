import numpy as np
from matplotlib import pyplot as plt

L = 2.533e-3
C = 1e-5
R = 20


def Atf(omega, L, C, R):
    return 1 / (1 + (1j * omega * R * C) / (1 - omega**2 * L * C))


omegas = np.logspace(0, 8, 200, base=10)
As = np.log(np.abs(Atf(omegas, L, C, R)))
# # As2 = np.abs(Atf(omegas, L, C, R))
# # As3 = AbsAtf(omegas, L, C, R)

fig, ax = plt.subplots()

ax.plot(omegas, As)
# # ax.plot(omegas, As2)
# # ax.plot(omegas, As3)
ax.set_xscale('log')
plt.grid()

# # ax.plot(omegas, np.angle(Atf(omegas, L, C, R)))

# plt.show()

omega_3dB1 = (+(R * C) + np.sqrt((R * C)**2 + 4 * L * C)) / (2 * L * C)
omega_3dB2 = (-(R * C) + np.sqrt((R * C)**2 + 4 * L * C)) / (2 * L * C)

print(omega_3dB1, omega_3dB2)
print(omega_3dB1 / (2 * np.pi), omega_3dB2 / (2 * np.pi))
print()
# empirical
Q_1 = 2000 * np.pi / (omega_3dB1 - omega_3dB2)
print(Q_1)

w_0 = 1 / np.sqrt(C * L)
print(w_0, w_0 / (2 * np.pi))
delta_w = R / L
print(delta_w, delta_w / (2 * np.pi))
print(w_0 / delta_w)
print(np.sqrt(L / C) / R)
print(1000 / (1809 - 552))
