"""Independent Fourier-collocation check of the generalized-Hopf amplitude law.

This code does NOT import or call the shooting, Poincare-Lindstedt,
or Floquet routines contained in the preceding manuscript package.
It substitutes a Fourier ansatz into the scalar jerk ODE and solves the
full harmonic-balance residual numerically for omega and k_s at fixed
fundamental cosine amplitude A. A (unlike peak-to-peak amplitude) is the
first-harmonic coefficient, matching the proof's normalization.
"""
import numpy as np
import pandas as pd
from scipy.optimize import root
from pathlib import Path
HERE=Path(__file__).resolve().parent
N=8; M=256
phi=np.linspace(0,2*np.pi,M,endpoint=False)
ksH=2.25
Lam=np.sqrt(290871/57856)
Om=np.sqrt(3.5)
m4=132620389936/72604730457
h2=-8/8379
h4=1574439536/508233113199
j=np.arange(2,N+1)
cosn=np.cos(np.outer(j,phi));sinn=np.sin(np.outer(j,phi))

def calc(u,A, return_full=False):
    c0=u[0]; ac=u[1:N]; bs=u[N:2*N-1]; omega=u[-2]; ks=u[-1]
    x=c0+A*np.cos(phi)+ac@cosn+bs@sinn
    dx=-A*np.sin(phi)+(-j*ac)@sinn+(j*bs)@cosn
    d2x=-A*np.cos(phi)+(-(j*j)*ac)@cosn+(-(j*j)*bs)@sinn
    d3x=A*np.sin(phi)+((j**3)*ac)@sinn+(-(j**3)*bs)@cosn
    y=(ks*x-x*x)/Lam
    F=Lam*np.tanh(y)
    res=omega**3*d3x+0.5*omega**2*d2x+3.5*omega*dx+4*x-F
    coeff=2/M*np.fft.rfft(res)[:N+1]
    out=np.r_[np.real(coeff[0])/2, np.real(coeff[1:]), -np.imag(coeff[1:])]
    if return_full:
        return np.max(np.abs(res)), out, x
    return out

rows=[]
for A in [.24,.20,.16,.12,.10,.08,.06,.045,.035]:
    u=np.zeros(2*N+1)
    u[0]=-2/7*A*A
    u[1]=2/1197*A*A
    u[N]=8*np.sqrt(3.5)/1197*A*A
    u[-2]=Om*(1+h2*A*A+h4*A**4)
    u[-1]=ksH+m4*A**4
    sol=root(calc,u,args=(A,),method='hybr',options={'xtol':1e-11,'maxfev':5000})
    maxres,coef,_=calc(sol.x,A,True)
    if max(abs(coef))>3e-10:raise RuntimeError('Nonconverged harmonic solution',A,sol.message, max(abs(coef)))
    omega,ks=sol.x[-2:]
    rows.append(dict(A=A,mu=ks-ksH,mu_over_A4=(ks-ksH)/A**4,
                     m4_exact=m4,relative_m4_error=abs((ks-ksH)/A**4/m4-1),
                     omega=omega,h2_pred=h2,delta_omega=(omega-Om),
                     max_pointwise_ode_residual=maxres,max_Fourier_residual=max(abs(coef)),success=sol.success))
frame=pd.DataFrame(rows)
frame.to_csv(HERE/'independent_fourier_collocation.csv',index=False,float_format='%.15g')
print(frame[['A','mu_over_A4','relative_m4_error','max_Fourier_residual']].to_string(index=False))
