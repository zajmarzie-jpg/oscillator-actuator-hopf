"""Independent periodic mechanical power/force-limit audit (Revision 15).

Evaluates 12 nondegenerate local Hopf cycles plus 8 Bautin-threshold
and 27 Lambda=2.3 cycles, using ONLY saved numerical periodic initial
states from the prior calculations and an independently defined RHS.
Does not call the existing shooting/normal-form implementations.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, simpson
ROOT = Path(__file__).resolve().parent
cases=[]
for name in ['saturated_hopf_criticality.csv','generalized_hopf_shooting.csv']:
    d=pd.read_csv(ROOT/name)
    for _,r in d.iterrows():
        cases.append((name,r))
records=[]
for filename,r in cases:
    lam=float(r['Lambda']); sigma=float(r['sigma']); ks=float(r['ks']); T=float(r['T'])
    b=sigma-1.0  # tau=1, omega0=2, d=0.2; g=d+1-sigma
    x0=float(r['x0']); f0=float(r['f0'])
    def rhs(t,y):
        x,v,f=y
        target=lam*np.tanh((ks*x-x*x)/lam)
        return [v,-4*x-b*v+f,target-f]
    sol=solve_ivp(rhs,[0,T],[x0,0.0,f0],method='DOP853',rtol=2e-12,atol=3e-14,dense_output=True)
    if not sol.success: raise RuntimeError(sol.message)
    t=np.linspace(0,T,8193)
    x,v,f=sol.sol(t)
    avg=lambda y:simpson(y,x=t)/T
    v_rms=np.sqrt(avg(v*v)); x_rms=np.sqrt(avg((x-avg(x))**2))
    power=avg(v*f); rate_diss=b*avg(v*v)
    total_variation=simpson(abs(v),x=t)
    excursion=np.max(x)-np.min(x)
    rms_bound=lam/abs(b)
    excursion_bound=T*lam/(2*abs(b))
    records.append(dict(dataset=filename,Lambda=lam,sigma=sigma,b=b,ks=ks,T=T,
         x_rms=x_rms,v_rms=v_rms,available_force=lam,
         actuator_force_peak=float(np.max(abs(f))),
         necessary_minimum_force=abs(b)*v_rms,
         force_utilization_ratio=abs(b)*v_rms/lam,
         displacement_excursion=excursion,
         excursion_upper_bound=excursion_bound,
         excursion_utilization_ratio=excursion/excursion_bound,
         time_averaged_actuator_power=power,
         time_averaged_damping_power=rate_diss,
         power_balance_residual=power-rate_diss,
         total_variation=total_variation,
         endpoint_closure_error=float(np.max(abs(sol.y[:,-1]-[x0,0,f0])))))
res=pd.DataFrame(records)
res.to_csv(ROOT/'force_budget_audit.csv',index=False,float_format='%.14g')
assert (res.force_utilization_ratio<1.+1e-9).all()
assert (res.excursion_utilization_ratio<1.+1e-9).all()
assert (res.actuator_force_peak <= res.Lambda+1e-10).all()
assert res.power_balance_residual.abs().max()<1e-8
assert res.endpoint_closure_error.max()<3e-8
print('periodic solutions checked:',len(res))
print('max power-balance residual:',res.power_balance_residual.abs().max())
print('max utilization ratios:',res.force_utilization_ratio.max(),res.excursion_utilization_ratio.max())
print('max closure:',res.endpoint_closure_error.max())
print(res.groupby(['dataset','Lambda']).agg(n=('Lambda','size'),max_vel=('v_rms','max'),max_ratio=('force_utilization_ratio','max'),max_abs_balance=('power_balance_residual',lambda s: np.max(abs(s)))).to_string())
