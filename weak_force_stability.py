#!/usr/bin/env python3
"""Independent global-convergence benchmark for Revision 10.
This script evaluates the Lyapunov certificate for the saturated quadratic command and integrates five
initial conditions. It does not substitute numerical tests for proof.
"""
from pathlib import Path
import csv
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
omega, tau, b, k_s, kappa, Lambda = 2.0, 1.0, 0.5, 0.0, 1.0, 0.01
L_cert = 4*np.exp(-0.5)*np.sqrt(abs(kappa)*Lambda)
A = np.array([[0, 1, 0], [-omega**2, -b, 1], [0, 0, -1/tau]], dtype=float)
P = solve_continuous_lyapunov(A.T, -np.eye(3))
assert np.allclose(A.T@P+P@A, -np.eye(3), atol=1e-12)
assert np.linalg.eigvalsh(P).min() > 0
p_e3 = np.linalg.norm(P[:, 2])
Lcrit = min(omega**2, tau/(2*p_e3))
eta = 1 - 2*L_cert*p_e3/tau
rate = eta/np.linalg.eigvalsh(P).max()
assert L_cert < Lcrit and eta > 0

initials = [
    (1., 0., 0.), (0., 2., -1.), (-2., 1., 3.),
    (3., -1., -2.), (-0.3, 0.4, 0.2)
]
t = np.linspace(0, 40, 2001)
bound = np.exp(-rate*t)
records = []
fig, ax = plt.subplots(figsize=(8.3, 4.6))
for i, initial in enumerate(initials, start=1):
    def ode(time, state):
        return A@state + np.array([0., 0., (Lambda*np.tanh((k_s*state[0]-kappa*state[0]**2)/Lambda))/tau])
    sol = solve_ivp(ode, (t[0], t[-1]), initial, t_eval=t,
                    method='DOP853', rtol=1e-11, atol=1e-13)
    assert sol.success
    V = np.einsum('it,ij,jt->t', sol.y, P, sol.y)
    ratio = V/V[0]
    excess = np.max(ratio-bound)
    assert excess < 1e-11, (initial, excess)
    ax.semilogy(t, ratio, linewidth=1.25,
                label=f'Initial condition {i}')
    records.append({'case':i, 'x0':initial[0], 'v0':initial[1],
                    'f0':initial[2], 'state_norm_t40':np.linalg.norm(sol.y[:,-1]),
                    'energy_ratio_t40':ratio[-1],
                    'maximum_upper_bound_excess':excess,
                    'certified_rate':rate, 'lipschitz_L_bound':L_cert,
                    'critical_L':Lcrit, 'eta':eta})
ax.semilogy(t, bound, 'k--', linewidth=2.0,
            label='Analytical upper bound')
ax.set_xlabel('Time')
ax.set_ylabel(r'$V(t)/V(0)$')
ax.set_title('Certified Lyapunov decay for the saturated quadratic actuator')
ax.set_ylim(1e-11, 1.6)
ax.grid(alpha=0.22)
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(HERE/'weak_force_decay.png', dpi=210)
plt.close(fig)
with (HERE/'weak_force_stability_results.csv').open('w', newline='') as f:
    wr=csv.DictWriter(f, fieldnames=records[0].keys())
    wr.writeheader()
    wr.writerows(records)
print('Lyapunov P=\n',P)
print('Lcrit', Lcrit, 'certified L', L_cert, 'eta', eta, 'rate',rate)
print('max upper bound excess:', max(d['maximum_upper_bound_excess'] for d in records))
