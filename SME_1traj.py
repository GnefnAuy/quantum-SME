#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:47:20 2026

@author: yuanfeng007
"""



import numpy as np
from qutip import *
import matplotlib.pyplot as plt
#parameters 
N = 5
wr = 1
wq = 1
chi = 0.05      # normally much smaller than wr and wq
ep = 0.2 + 0j
phi = np.pi / 4
eta = 0.8
kappa = 1
gamma = 0.01
gamma_phi = 0.01


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



sresult = smesolve(H, rho0, tlist, c_ops=c_ops, sc_ops=sc_ops, e_ops=e_ops,
                  ntraj=1,          # one conditioned stochastic trajectory
                    seeds=0,
                    options=options
                  )


# Conditional trajectory results
states = sresult.states

e_ops = [n, sx, sy, sz, X_phi]
yn, yx, yy, yz, yphi = sresult.expect

# Homodyne measurement
signal = np.sqrt(eta * kappa) * np.real(yphi[0][1:])
record = np.asarray(sresult.measurement)[0, 0].real
noise = record - signal
mtimes = tlist[1:]


fig,ax = plt.subplots(1,3,figsize=(12,4))
ax = ax.ravel()
ax[0].plot(mtimes,signal)
ax[0].set_ylabel(r"$signal\langle X_\phi\rangle$")
ax[0].set_xlabel("t")

ax[1].plot(mtimes,record)
ax[1].set_ylabel("noisy homodyne-detector output")
ax[1].set_xlabel("t")

ax[2].plot(mtimes,noise)
ax[2].set_ylabel("noise")
plt.suptitle("Obervables evolving with SME")
fig.tight_layout()
plt.show()

