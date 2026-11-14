"""Independent bounded-force/anti-damping escape audit for Revision 13.

Compare finite-bandwidth tanh-saturated actuator trajectories to the
mechanical homogeneous system. This is a numerical illustration, not a
replacement for the analytical proposition in Active.tex.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import expm

out=Path(__file__).resolve().parent
omega=2.0; tau=1.0; b=-0.5; kappa=1.0; Lambda=2.3
ks=2.25-8e-5
H=np.array([[0.,1.],[-omega**2,-b]])
B=np.array([0.,1.])

def fun(t,u):
    x,v,f=u
    cmd=Lambda*np.tanh((ks*x-kappa*x*x)/Lambda)
    return [v,-omega**2*x-b*v+f,(cmd-f)/tau]

records=[]
for x0 in [1.0,2.0,4.0,8.0]:
    f0=0.
    sol=solve_ivp(fun,(0,28),[x0,0,f0],rtol=2e-11,atol=3e-12,max_step=0.03,dense_output=True)
    if not sol.success: raise RuntimeError(sol.message)
    vals=sol.sol(np.linspace(0,28,2001))
    maxf=np.max(abs(vals[2])); fbound=max(abs(f0),Lambda)
    mech_initial=np.linalg.norm(vals[:2,0]); mech_last=np.linalg.norm(vals[:2,-1]);
    # For an unstable structural matrix, every trajectory with sufficiently
    # large initial mechanical norm has nonzero unstable asymptotic datum:
    # z_inf=z(0)+int_0^inf exp(-Hs) B f(s) ds.
    # We only integrate a finite window here and report its partial value.
    from scipy.integrate import trapezoid
    ts=np.linspace(0,28,2001)
    integrand=np.stack([expm(-H*t)@B * f for t,f in zip(ts,vals[2])])
    winf=np.array([x0,0.])+trapezoid(integrand,ts,axis=0)
    records.append(dict(initial_x=x0,initial_v=0.,initial_f=f0,final_t=28,
                        final_x=vals[0,-1],final_v=vals[1,-1],final_f=vals[2,-1],
                        mechanical_norm_initial=mech_initial,mechanical_norm_final=mech_last,
                        max_force_abs=maxf,force_bound=fbound,partial_asymptotic_vector_norm=np.linalg.norm(winf),
                        solver_status=sol.status,nfev=sol.nfev))
assert all(r['max_force_abs'] <= Lambda+1e-9 for r in records)
assert all(r['mechanical_norm_final']>r['mechanical_norm_initial']*25 for r in records)
pd.DataFrame(records).to_csv(out/'escape_audit.csv',index=False,float_format='%.13g')
print(pd.DataFrame(records).to_string(index=False))
