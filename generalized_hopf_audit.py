"""Independent exact fifth-order Poincare-Lindstedt audit and full-system shooting.

Specified parameter point: omega0=2, tau=kappa=1, sigma=1/2, ks_H=9/4.
Complex Fourier arithmetic is performed in Q(i*sqrt(7/2)); no numerical
fitting is used to obtain the quintic coefficient. The 3D shooting
verification, Floquet multipliers and cycle fold are separate numerical tests.
Run from this directory: python generalized_hopf_audit.py
"""
from __future__ import annotations
import json
from collections import defaultdict
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import root
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from step3_analysis import residual_jac,characterize

OUT=Path(__file__).resolve().parent
W2=Q(7,2); SIG=Q(1,2); KH=Q(9,4);KAPPA=Q(1);TAU=Q(1)
LAM2=KH**3*W2*SIG*(4*W2+SIG**2)/(2*KAPPA**2*(8*W2+SIG**2))

class C:
    """Exact a + (i sqrt(W2)) b, with a,b rational."""
    def __init__(self,a=0,b=0):
        if isinstance(a,C): self.a,self.b=a.a,a.b
        else: self.a,self.b=Q(a),Q(b)
    def __add__(self,other):
        other=C(other);return C(self.a+other.a,self.b+other.b)
    __radd__=__add__
    def __neg__(self):return C(-self.a,-self.b)
    def __sub__(self,other):return self+-C(other)
    def __mul__(self,other):
        other=C(other);return C(self.a*other.a-W2*self.b*other.b,self.a*other.b+self.b*other.a)
    __rmul__=__mul__
    def __truediv__(self,other):
        other=C(other)
        det=other.a**2+W2*other.b**2
        return C((self.a*other.a+W2*self.b*other.b)/det,(self.b*other.a-self.a*other.b)/det)
    def __bool__(self):return bool(self.a or self.b)

def Z(n):return C(0,n)
def L0(n):
    z=Z(n);return (z+SIG)*(z*z+W2)
def L1(n):
    z=Z(n);return z*(3*z*z+2*SIG*z+W2)
def L2(n):
    z=Z(n);return z*z*(3*z+SIG)

# Taylor series of Lambda tanh((ks*x-kappa*x*x)/Lambda) at critical ks.
coeff={2:-KAPPA,3:-KH**3/(3*LAM2),4:KH**2*KAPPA/LAM2,
       5:-KH*KAPPA**2/LAM2+2*KH**5/(15*LAM2**2)}
X={(1,1):C(Q(1,2)),(1,-1):C(Q(1,2))}
def mult(a,b):
    out=defaultdict(C)
    for (i,h),v in a.items():
        for (j,k),w in b.items():
            if i+j<=5:out[(i+j,h+k)]+=v*w
    return dict(out)
def nonlin(order):
    out=defaultdict(C); power=dict(X)
    for j in range(2,6):
        power=mult(power,X)
        for (degree,h),v in power.items():
            if degree==order:out[h]+=coeff[j]*v
    return out

h2=h4=m4=None
for order in range(2,6):
    nl=nonlin(order)
    if order==3:
        h2=nl[1].b/(L1(1).b*Q(1,2))
        secular=L1(1)*h2*Q(1,2)-nl[1]
        assert secular.a==secular.b==0, 'nonzero third-order real resonance'
    if order==5:
        known=L2(1)*h2*h2*Q(1,2)-nl[1]
        h4=-known.b/(L1(1).b*Q(1,2))
        m4=2*(known+L1(1)*h4*Q(1,2)).a
    for h in range(-order,order+1):
        if (order+h)%2 or abs(h)==1:continue
        residual=-nl[h]
        if order>=3:residual+=L1(h)*h2*X.get((order-2,h),C())
        if order>=5:residual+=L2(h)*h2*h2*X.get((order-4,h),C())
        if residual:X[(order,h)]=-residual/L0(h)

assert LAM2==Q(290871,57856)
assert h2==Q(-8,8379)
assert h4==Q(1574439536,508233113199)
assert m4==Q(132620389936,72604730457)
alpha=-Q(1)/(2*(W2+SIG*SIG))
gamma=-alpha*m4
assert gamma==Q(265240779872,1089070956855)
Lamcrit=np.sqrt(float(LAM2));kh=float(KH);sigma=float(SIG)

# Fixed-turning-point shooting: solve periodic closure for (f0,T,ks)
# at a prescribed displacement maximum x(0)=x0, v(0)=0.
records=[]
def solve_at_x0(x0,Lambda_,qseed):
    def rf(u):return residual_jac(np.r_[x0,u],sigma,Lambda_)[0]
    def jf(u):return residual_jac(np.r_[x0,u],sigma,Lambda_,sens=True)[1][:,1:4]
    ans=root(rf,qseed,jac=jf,tol=1e-10)
    if max(abs(rf(ans.x)))>2e-8:raise RuntimeError(f'shooting failure x0={x0}')
    q=np.r_[x0,ans.x]
    data=characterize(q,sigma,Lambda_)
    if data['amplitude']<1e-4:raise RuntimeError('converged to equilibrium')
    return ans.x, data

for Lambda_,grid in [(Lamcrit,[.35,.28,.23,.19,.16,.13,.10,.075]),
                     (2.3,list(np.linspace(.26,.045,27)))]:
    qseed=np.array([.5*grid[0],2*np.pi/np.sqrt(float(W2)),kh+float(m4)*grid[0]**4])
    for x0 in grid:
        qseed,data=solve_at_x0(x0,Lambda_,qseed)
        data.update(Lambda=Lambda_,mu=data['ks']-kh,
                    mu_over_A4=(data['ks']-kh)/data['amplitude']**4)
        records.append(data)
df=pd.DataFrame(records)
df.to_csv(OUT/'generalized_hopf_shooting.csv',index=False,float_format='%.14g')

# Locate saddle-node of cycles as extremum of ks along the fixed-x0
# family: implicit derivative of (f0,T,ks) with respect to x0.
def fold_res(q):
    r,J=residual_jac(q,sigma,2.3,sens=True)
    deriv=np.linalg.solve(J[:,1:4],-J[:,0])
    return np.r_[r,deriv[-1]]
g=df[df.Lambda==2.3].copy()
nearest=g.iloc[(g.mu-g.mu.min()).abs().argmin()]
q0=np.array([nearest["x0"],nearest["f0"],nearest["T"],nearest["ks"]])
ans=root(fold_res,q0,tol=1e-9)
if max(abs(fold_res(ans.x)))>1e-7:raise RuntimeError('fold solve failed')
fold=characterize(ans.x,sigma,2.3)
fold.update(Lambda=2.3,mu=fold['ks']-kh,detected_fold=True,derivative_residual=float(fold_res(ans.x)[-1]))
pd.DataFrame([fold]).to_csv(OUT/'generalized_hopf_fold.csv',index=False,float_format='%.15g')
assert abs(fold['floquet_modulus']-1)<3e-5
assert (df.shooting_residual.abs()<2e-8).all()
assert (df[df.Lambda==Lamcrit].floquet_modulus>1).all()

# Combine previous numerical criticality result and new Bautin data
older=pd.read_csv(OUT/'saturated_hopf_criticality.csv')
fig,axs=plt.subplots(1,2,figsize=(10.4,3.8))
for lam,marker,lab in [(1.5,'o','Subcritical, $\\Lambda=1.5$'),(5.,'s','Supercritical, $\\Lambda=5$')]:
    rows=older[older.Lambda==lam].sort_values('mu')
    axs[0].scatter(abs(rows.mu),rows.amplitude**2,s=20,marker=marker,label=lab)
    axs[0].plot(abs(rows.mu),(-rows.alpha.iloc[0]*np.sign(rows.beta.iloc[0])/rows.beta.iloc[0])*abs(rows.mu),ls='--',lw=1.1)
axs[0].set(xlabel=r'$|\mu|$',ylabel=r'$A^2$',title='Ordinary Hopf on either side')
axs[0].legend(fontsize=7)

u=g.sort_values('amplitude')
colors=np.where(u.floquet_modulus.to_numpy()>1+1e-6,'unstable','attracting')
for typ,marker in [('unstable','o'),('attracting','s')]:
    z=u.loc[colors==typ]
    axs[1].scatter(z['mu'],z['amplitude'],marker=marker,s=22,label=typ.capitalize())
axs[1].plot(u.mu,u.amplitude,lw=.7,alpha=.6)
axs[1].scatter([fold['mu']],[fold['amplitude']],s=70,marker='*',label='Cycle fold')
axs[1].axvline(0,ls=':',lw=.8)
axs[1].set(xlabel=r'$\mu=k_s-k_{s,\mathrm{H}}$',ylabel=r'$A$',title=r'Bautin unfolding, $\Lambda=2.3$')
axs[1].set_xlim(-.00016,.0005)
axs[1].set_ylim(.035,.18)
axs[1].legend(fontsize=7)
for ax in axs:ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(OUT/'saturated_hopf_criticality.png',dpi=220)
plt.close(fig)

summary=dict(Lambda_crit_squared=str(LAM2),Lambda_crit=Lamcrit,
             h2=str(h2),h4=str(h4),m4=str(m4),m4_numeric=float(m4),
             gamma=str(gamma),gamma_numeric=float(gamma),
             fold_Lambda=2.3,fold_mu=float(fold['mu']),fold_amplitude=float(fold['amplitude']),
             fold_floquet_modulus=float(fold['floquet_modulus']),
             fold_residual=float(fold['shooting_residual']),
             count=int(len(df)),max_shoot_residual=float(df.shooting_residual.max()),
             smallest_critical_amplitude=float(df[df.Lambda==Lamcrit].amplitude.min()),
             critical_smallest_amplitude_ratio=float(df[df.Lambda==Lamcrit].sort_values('amplitude').iloc[0].mu_over_A4))
(OUT/'generalized_hopf_exact_results.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
