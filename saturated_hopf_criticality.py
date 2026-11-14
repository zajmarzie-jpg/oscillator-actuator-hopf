"""Independent normal-form-vs-shooting test of saturation-induced Hopf-criticality reversal.

Uses the original 3D oscillator-actuator ODE and integrates the full 3x3 variational
system. The analytical cubic coefficient is obtained from the fundamental,
mean and second harmonics of the scalar jerk equation. No curve fitting is
used to derive the analytical coefficient. Run in this directory:
  python saturated_hopf_criticality.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import root
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from step3_analysis import residual_jac, characterize

OUT=Path(__file__).resolve().parent
OMEGA0=2.;TAU=1.;SIGMA=.5;KAPPA=1.
OMEGA2=OMEGA0**2-TAU**-2+SIGMA/TAU
KS_H=OMEGA0**2-TAU*SIGMA*OMEGA2
ALPHA=-1./(2*TAU*(OMEGA2+SIGMA**2))
BETA_QUAD=-KAPPA**2*(8*OMEGA2+SIGMA**2)/(4*TAU**2*OMEGA2*SIGMA*(OMEGA2+SIGMA**2)*(4*OMEGA2+SIGMA**2))
LAMBDA_CRIT=np.sqrt(KS_H**3*TAU*OMEGA2*SIGMA*(4*OMEGA2+SIGMA**2)/(2*KAPPA**2*(8*OMEGA2+SIGMA**2)))

def coefficient(fmax):
 return BETA_QUAD+KS_H**3/(8*TAU*fmax**2*(OMEGA2+SIGMA**2))

def solve(fmax,mu):
 beta=coefficient(fmax)
 amp=np.sqrt(-ALPHA*mu/beta)
 q0=np.array([amp,.5*amp,2*np.pi/np.sqrt(OMEGA2)])
 def resid(q):return residual_jac(np.r_[q,KS_H+mu],SIGMA,fmax)[0]
 def jac(q):return residual_jac(np.r_[q,KS_H+mu],SIGMA,fmax)[1][:,:3]
 ans=root(resid,q0,jac=jac,tol=4e-11)
 err=np.max(np.abs(resid(ans.x)))
 if err>2e-8 or abs(ans.x[0])<1e-4:
  raise RuntimeError(f'nontrivial shooting failed Lambda={fmax}, mu={mu}, err={err}, x0={ans.x[0]}')
 row=characterize(np.r_[ans.x,KS_H+mu],SIGMA,fmax)
 if row['amplitude']<1e-4:raise RuntimeError('trivial orbit')
 row.update(mu=mu,Lambda=fmax,beta=beta,alpha=ALPHA,
            Hopf_ks=KS_H,Lambda_crit=LAMBDA_CRIT,
            predicted_amplitude=amp,
            relative_amplitude_error=(row['amplitude']/amp-1),
            amplitude_sq_to_mu=row['amplitude']**2/mu,
            expected_sq_to_mu=-ALPHA/beta)
 return row

if __name__=='__main__':
 print(f'sigma={SIGMA}, omega^2={OMEGA2}, ks_H={KS_H}, alpha={ALPHA}, beta_quadratic={BETA_QUAD}, Lambda_crit={LAMBDA_CRIT}',flush=True)
 rec=[]
 for FMAX in (1.5,5.):
  beta=coefficient(FMAX)
  sign=np.sign(beta)
  print('Lambda',FMAX,'beta',beta,flush=True)
  for m in (.001,.002,.003,.005,.007,.01):
   row=solve(FMAX,sign*m)
   rec.append(row)
   print('mu',row['mu'],'Afull',row['amplitude'],'Apred',row['predicted_amplitude'],'Floquet',row['floquet_modulus'],'closure',row['shooting_residual'],flush=True)
 df=pd.DataFrame(rec)
 df.to_csv(OUT/'saturated_hopf_criticality.csv',index=False,float_format='%.13g')
 fig,ax=plt.subplots(figsize=(7.6,3.2))
 specs=[(1.5,'o','Unstable branch, $\\Lambda=1.5$'),(5.,'s','Attracting branch, $\\Lambda=5$')]
 for f,marker,lab in specs:
  s=df[df.Lambda==f].sort_values('mu')
  x=abs(s.mu.values)
  beta=coefficient(f)
  ax.plot(x,(-ALPHA*np.sign(beta)/beta)*x,ls='--',lw=1.15,label='Leading Hopf prediction, $\\Lambda='+str(f)+'$')
  ax.scatter(x,s.amplitude.values**2,marker=marker,s=27,label=lab,zorder=3)
 ax.set(xlabel=r'$|k_s-k_{s,\mathrm{H}}|$',ylabel=r'Squared half peak-to-peak amplitude $A^2$',title='Classical Hopf criticality under actuator saturation')
 ax.grid(alpha=.2);ax.legend(fontsize=7.2,loc='best',ncol=2)
 fig.tight_layout();fig.savefig(OUT/'saturated_hopf_criticality.png',dpi=200);plt.close(fig)
 assert (df.shooting_residual<1e-7).all()
 assert (df[df.Lambda==1.5].floquet_modulus>1).all()
 assert (df[df.Lambda==5.].floquet_modulus<1).all()
 print('Records:',len(df),'Worst closure:',df.shooting_residual.max(), 'Min Floquet separation from unit:',min(abs(df.floquet_modulus-1)),flush=True)
