#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:35:24 2026

@author: yuanfeng007
"""

import numpy as np
from qutip import *
import matplotlib.pyplot as plt



#parameters 
N = 5
wr = 1
wq = 1
chi = 0.1      # normally much smaller than wr and wq
ep = 0.2 + 0j
phi = np.pi / 4
eta = 0.8
kappa = 1
gamma = 0.01
gamma_phi = 0.01


#change of parameters


chi_values = [0.05, 0.10, 0.15, 0.20]



tlist = np.linspace(0, 1000, 101)  # Output spacing: 1


sx = tensor(qeye(N),sigmax())
sy = tensor(qeye(N),sigmay())
sz = tensor(qeye(N),sigmaz())

sm = tensor(qeye(N),sigmam())

a = tensor(destroy(N),qeye(2))
n = a.dag() * a 
X_phi = np.exp(-1j*phi)*a + np.exp(1j*phi)*a.dag()

# Hamiltonian 
H = (wr * n + 0.5 * wq * sz 
     + chi * n * sz 
     + ep * a.dag() + ep.conjugate() *a)



# Stochastic Master Equations

options = {
    "dt": 0.001,
    "store_states": True,
    "store_measurement": True,
    "keep_runs_results": True
}      # Internal integration step: 0.001


rho0 = tensor(basis(N,0), (basis(2,0)+basis(2,1)).unit())
c_ops = [
    np.sqrt((1 - eta) * kappa) * a,
    np.sqrt(gamma) * sm,
    np.sqrt(gamma_phi / 2) * sz,  # if gamma_phi is the pure-dephasing rate
]

sc_ops = [
    np.sqrt(eta * kappa) * np.exp(-1j * phi) * a
]
e_ops = [n, sx, sy, sz, X_phi]

ntraj = 100

sresult = smesolve(H, rho0, tlist, c_ops=c_ops, sc_ops=sc_ops, e_ops=e_ops,
                  ntraj=ntraj,          # one conditioned stochastic trajectory
                    seeds=123,
                    options=options
                  )








# Each has shape (ntraj, len(tlist))
yn_runs = np.asarray(sresult.runs_expect[0]).real
yx_runs = np.asarray(sresult.runs_expect[1]).real
yy_runs = np.asarray(sresult.runs_expect[2]).real
yz_runs = np.asarray(sresult.runs_expect[3]).real
yphi_runs = np.asarray(sresult.runs_expect[4]).real

# Shape: (ntraj, len(sc_ops), len(tlist)-1)
measurements = np.asarray(sresult.measurement)

# First monitored channel:
# shape (ntraj, len(tlist)-1)
current_runs = measurements[:, 0, :].real

# Conditional mean and noise for every trajectory
signal_runs = np.sqrt(eta * kappa) * yphi_runs[:, 1:]
noise_runs = current_runs - signal_runs

print("Photon populations:", yn_runs.shape)
print("Signals:", signal_runs.shape)
print("Currents:", current_runs.shape)
print("Noises:", noise_runs.shape)




#distribution
time_average_list = np.mean(
    current_runs[:, 50:],
    axis=1,
)

data = time_average_list

mu = np.mean(data)
var = np.var(data)
std = np.std(data)

plt.figure()
plt.hist(data, bins=15, density=True, alpha=0.7)
plt.xlabel("Value")
plt.ylabel("Probability density")
# Mark mean
plt.axvline(mu, color="k", linestyle="--",
            linewidth=2, label=f"Mean = {mu:.4f}")

# Mark mean ± standard deviation
plt.axvline(mu - std, color="green", linestyle=":",
            label=r"Mean $\pm$ SD")
plt.axvline(mu + std, color="green", linestyle=":")

# Display variance in legend
plt.plot([], [], " ", label=f"Variance = {var:.6f}")

plt.xlabel("Time-averaged homodyne current")
plt.ylabel("Probability density")
plt.legend()
plt.title(r"Probabilty distribution of Record with $\chi=0.05$")
plt.savefig(
    "ProDist_chi005_SME.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()



#plt


mtimes = tlist[1:]
number_to_plot = min(5, ntraj)
colors = plt.cm.tab10(np.arange(number_to_plot))

# Average across trajectories, not across time
mean_signal = np.mean(signal_runs, axis=0)
mean_current = np.mean(current_runs, axis=0)
mean_noise = np.mean(noise_runs, axis=0)

fig, ax = plt.subplots(
    1, 3,
    figsize=(12, 4),
    sharex=True,
)

for i in range(number_to_plot):
    label = f"Trajectory {i + 1}"

    ax[0].plot(
        mtimes,
        signal_runs[i],
        color=colors[i],
        label=label,
    )

    ax[1].plot(
        mtimes,
        current_runs[i],
        color=colors[i],
        alpha=0.75,
        label=label,
    )

    ax[2].plot(
        mtimes,
        noise_runs[i],
        color=colors[i],
        alpha=0.75,
        label=label,
    )

# Add trajectory averages
ax[0].plot(
    mtimes,
    mean_signal,
    color="black",
    linewidth=3,
    label="Trajectory average",
    zorder=10,
)

ax[1].plot(
    mtimes,
    mean_current,
    color="black",
    linewidth=3,
    label="Trajectory average",
    zorder=10,
)

ax[2].plot(
    mtimes,
    mean_noise,
    color="black",
    linewidth=3,
    label="Trajectory average",
    zorder=10,
)



ax[0].set_title("Conditional Homodyne Signal")
ax[1].set_title("Homodyne Measurement Record")
ax[2].set_title("Measurement Innovation Noise")

ax[0].set_ylabel(r"$\sqrt{\eta\kappa}\,\langle X_\phi\rangle_c$")
ax[1].set_ylabel(r"$J_\phi(t)$")
ax[2].set_ylabel(r"$J_\phi(t)-\sqrt{\eta\kappa}\langle X_\phi\rangle_c$")

ax[0].legend(
        title="SME trajectories",
        fontsize=7,
        title_fontsize=8,
        loc="best",
        frameon=True,
    )

for axis in ax:
    axis.set_xlabel(r"Time, $t$")
    axis.grid(alpha=0.25)


fig.suptitle(
    "Stochastic Homodyne Measurement Dynamics",
    fontsize=14,
    fontweight="bold",
)

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.tight_layout()

fig.savefig(
    "signalrecord_SME_chi010.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
