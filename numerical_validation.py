"""Reproducible numerical check for the user's oscillator--actuator manuscript.
Author-independent validation script; numerical experiments do not constitute physical validation.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

w0,tau,kappa,d,g1,k1=2.0,1.0,1.0,0.2,-0.2,1.0
O=np.sqrt(w0*w0-1/tau**2)
r0=np.sqrt((k1*k1-g1*g1*O**4*tau**2)/(2*kappa*kappa))
u0=w0*w0/(2*kappa)*(k1-g1*O*O*tau)

def run_zh(eps):
    g=d+1/tau+g1*eps;ks=w0*w0+k1*eps
    def rhs(t,y):
        x,v,f=y[:3]
        z=np.array([v,-w0*w0*x+(g-d)*v+f,(ks*x-f-kappa*x*x)/tau])
        if len(y)==3:return z
        A=np.array([[0,1,0],[-w0*w0,g-d,1],[(ks-2*kappa*x)/tau,0,-1/tau]])
        return np.r_[z,(A@y[3:].reshape(3,3)).ravel()]
    def periodicity(z):
        x0,f0,T=z
        if T<=0:return np.ones(3)*1000
        sol=solve_ivp(rhs,(0,T),[x0,0,f0],method='DOP853',rtol=3e-12,atol=3e-14)
        x,v,f=sol.y[:,-1]
        return np.array([x-x0,v,f-f0])
    xinit=eps*(r0+u0/w0**2);finit=eps*(r0/tau**2+u0)
    ini=np.array([xinit,finit,2*np.pi/O])
    sr=root(periodicity,ini,tol=2e-9)
    resid=periodicity(sr.x)
    if max(abs(resid))>1e-9:
        print('bad solve',eps,sr.message,sr.x,resid)
        return None
    x0,f0,T=sr.x
    yinit=np.r_[x0,0,f0,np.eye(3).ravel()]
    sol=solve_ivp(rhs,(0,T),yinit,method='DOP853',rtol=3e-12,atol=3e-14,dense_output=True)
    eig=np.linalg.eigvals(sol.y[3:,-1].reshape(3,3))
    # exclude trivial multiplier numerically
    ix=np.argmin(abs(eig-1)); nontriv=np.delete(eig,ix)
    t=np.linspace(0,T,1501); trajectory=sol.sol(t)[:3,:];x=trajectory[0]
    rad=.5*(x.max()-x.min());mean=x.mean();expected_amp=abs(eps)*r0
    return dict(epsilon=eps,x0=x0,f0=f0,T=T,amp=rad,pred_amp=expected_amp,rel_amp_error=(rad-expected_amp)/expected_amp,mean_x=mean,pred_mean=eps*u0/w0**2,nontriv=nontriv,det=np.linalg.det(sol.y[3:,-1].reshape(3,3)),expecteddet=np.exp(eps*g1*T),success=sr.success)

if __name__=='__main__':
    print('zero Hopf:', 'r0',r0,'u0',u0,'Omega',O)
    for e in [0.005,0.01,0.02,0.04,0.08]:
        print(run_zh(e))

def run_hopf(eps, sigma=0.5, eta2=-1.0):
    O2=w0*w0-1/tau**2+sigma/tau; O=np.sqrt(O2)
    D=O2+sigma**2
    RH=np.sqrt(-2*tau*O2*sigma*(4*O2+sigma**2)*eta2/(kappa*kappa*(8*O2+sigma**2)))
    g=d+1/tau-sigma
    ks=w0*w0-tau*sigma*O2+eta2*eps*eps
    def rhs(t,y):
        x,v,f=y[:3]
        z=np.array([v,-w0*w0*x+(g-d)*v+f,(ks*x-f-kappa*x*x)/tau])
        if len(y)==3:return z
        A=np.array([[0,1,0],[-w0*w0,g-d,1],[(ks-2*kappa*x)/tau,0,-1/tau]])
        return np.r_[z,(A@y[3:].reshape(3,3)).ravel()]
    def periodicity(z):
        x0,f0,T=z
        if T<=0:return np.ones(3)*1000
        sol=solve_ivp(rhs,(0,T),[x0,0,f0],method='DOP853',rtol=3e-12,atol=3e-14)
        x,v,f=sol.y[:,-1]
        return np.array([x-x0,v,f-f0])
    init=np.array([eps*RH,eps*RH*(1-sigma*tau)/tau**2,2*np.pi/O])
    sr=root(periodicity,init,tol=5e-10)
    resid=periodicity(sr.x)
    if max(abs(resid))>1e-8 or abs(sr.x[0])<eps*RH*0.2:
        print('bad Hopf solve',eps,sr.message,sr.x,resid)
        return None
    x0,f0,T=sr.x
    sol=solve_ivp(rhs,(0,T),np.r_[x0,0,f0,np.eye(3).ravel()],method='DOP853',rtol=3e-12,atol=3e-14,dense_output=True)
    eig=np.linalg.eigvals(sol.y[3:,-1].reshape(3,3)); nontriv=np.delete(eig,np.argmin(abs(eig-1)))
    x=sol.sol(np.linspace(0,T,1501))[0,:]
    rad=(x.max()-x.min())/2
    radialexp=eta2*eps**2/(tau*D)
    return dict(epsilon=eps,ks=ks,x0=x0,T=T,amp=rad,pred_amp=eps*RH,rel_amp_error=(rad-eps*RH)/(eps*RH),nontriv=nontriv,radialexp_pred=radialexp)

if __name__=='__main__':
    print('classical Hopf sigma=0.5 eta2=-1:')
    for e in [0.02,0.04,0.08,0.12]: print(run_hopf(e))
    print("unstable classical Hopf sigma=-0.3 eta2=0.7:",run_hopf(0.02,sigma=-0.3,eta2=0.7))
