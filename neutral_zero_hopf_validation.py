"""Targeted reproducibility check: conservative (g1=0) zero-Hopf branch.
The periodic orbit is corrected by shooting in the FULL three-state system,
and multipliers come from the FULL variational system.
"""
import csv
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

W0=2.; TAU=1.; KAP=1.; D=0.2; K1=1.
OMEGA=np.sqrt(W0**2-1/TAU**2)

def sample(epsilon):
    g=D+1/TAU; ks=W0*W0+K1*epsilon
    def ode(t,z):
        x,v,f=z[:3]
        val=[v,-W0*W0*x+(g-D)*v+f,(ks*x-f-KAP*x*x)/TAU]
        if len(z)==3:return val
        J=np.array([[0.,1.,0.],[-W0*W0,g-D,1.],[(ks-2*KAP*x)/TAU,0.,-1/TAU]])
        return np.r_[val,(J@z[3:].reshape(3,3)).ravel()]
    r0=np.sqrt(K1*K1/(2*KAP*KAP))
    u0=W0*W0*K1/(2*KAP)
    q0=np.array([epsilon*(r0+u0/W0**2),epsilon*(r0/TAU**2+u0),2*np.pi/OMEGA])
    def residual(q):
        x0,f0,T=q
        if T<0.25:return np.ones(3)*1e5
        sol=solve_ivp(ode,(0,T),[x0,0.,f0],rtol=1e-12,atol=1e-14,method='DOP853')
        return sol.y[:,-1]-[x0,0.,f0]
    ret=root(residual,q0,tol=2e-10)
    assert max(abs(residual(ret.x)))<3e-10
    x0,f0,T=ret.x
    sol=solve_ivp(ode,(0,T),np.r_[x0,0,f0,np.eye(3).ravel()],rtol=1e-12,atol=1e-14,method='DOP853',dense_output=True)
    M=sol.y[3:,-1].reshape(3,3)
    lamb=np.linalg.eigvals(M); pair=np.delete(lamb,np.argmin(abs(lamb-1)))
    assert abs(np.linalg.det(M)-1)<1e-9
    assert abs(pair[0]-np.conj(pair[1]))<1e-8
    assert max(abs(abs(pair)-1))<1e-8
    ts=np.linspace(0,T,1800,endpoint=False)
    x=sol.sol(ts)[0,:]
    return dict(epsilon=epsilon,T=T,amp=(x.max()-x.min())/2,amplitude_firstorder=abs(epsilon)*r0,
                multiplier_re=pair[0].real,multiplier_im_abs=abs(pair[0].imag),
                max_multiplier_modulus_error=max(abs(abs(pair)-1)),monodromy_det=np.linalg.det(M),
                shooting_residual=max(abs(residual(ret.x))))

if __name__=='__main__':
    rows=[sample(e) for e in [.03,.05,.08,.12]]
    with open('neutral_zero_hopf_validation.csv','w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    for r in rows: print(r)
