"""Parameter continuation with direct full-system shooting and global identity checks.

This is fixed-parameter continuation, not pseudo-arclength: folds may be missed.
Requires scipy, numpy, pandas, matplotlib. Reproduces exact mean/energy relations
and nontrivial Floquet multipliers at representative stable/unstable branches.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt

W0=2.0; TAU=1.0; KAPPA=1.0; D=0.2
RTOL=2e-11; ATOL=4e-13

def setting(sigma,ks):
    g=D+1/TAU-sigma
    return g, W0**2+(D-g)/TAU

def rhs(t,z,sigma,ks):
    x,v,f=z[:3];g,W2=setting(sigma,ks)
    F=np.array([v,-W0**2*x+(g-D)*v+f,(ks*x-f-KAPPA*x*x)/TAU])
    if len(z)==3:return F
    J=np.array([[0.,1.,0.],[-W0**2,g-D,1.],[(ks-2*KAPPA*x)/TAU,0.,-1./TAU]])
    return np.r_[F,(J@z[3:].reshape(3,3)).ravel()]

def integrate_cycle(x0,f0,T,sigma,ks,dense=False):
    y=np.r_[x0,0.,f0,np.eye(3).ravel()]
    return solve_ivp(rhs,(0,T),y,args=(sigma,ks),method='DOP853',rtol=RTOL,atol=ATOL,dense_output=dense)

def correct_cycle(guess,sigma,ks):
    # shoot for a displacement turning point, and exploit variational Jacobian
    def calc(q, want_jac):
        x0,f0,T=q
        if T<.15 or T>100 or np.max(np.abs(q))>1e4:
            return (np.ones(3)*1e4 if not want_jac else np.eye(3)*1e4)
        s=integrate_cycle(x0,f0,T,sigma,ks)
        if not s.success: raise RuntimeError(s.message)
        ye=s.y[:3,-1]; M=s.y[3:,-1].reshape(3,3)
        if not want_jac:return ye-np.array([x0,0,f0])
        J=np.column_stack([M[:,0]-[1.,0.,0.],M[:,2]-[0.,0.,1.],rhs(T,ye,sigma,ks)])
        return J
    sol=root(lambda q:calc(q,False),guess,jac=lambda q:calc(q,True),tol=3e-10)
    q=sol.x
    if max(abs(calc(q,False)))>3e-8 or not (0.2<q[2]<10) or not np.all(np.isfinite(q)):
        return None
    # exclude stationary roots and incorrect min turning point
    if abs(q[0])<1e-7 or rhs(0,np.r_[q[0],0,q[1]],sigma,ks)[1]>=0: return None
    return q

def characterize(q,sigma,ks):
    x0,f0,T=q; S=integrate_cycle(x0,f0,T,sigma,ks,dense=True)
    M=S.y[3:,-1].reshape(3,3)
    grid=np.linspace(0,T,1001,endpoint=False)
    traj=S.sol(grid)[:3]
    x,v,f=traj
    amp=(x.max()-x.min())/2
    mu=np.mean(x); x2=np.mean(x*x); vsq=np.mean(v*v)
    acc=-W0**2*x+(setting(sigma,ks)[0]-D)*v+f
    accsq=np.mean(acc*acc)
    eig=np.linalg.eigvals(M)
    eig=np.delete(eig,np.argmin(abs(eig-1)))
    g,W2=setting(sigma,ks);delta=ks-W0**2
    defectmean=KAPPA*x2-delta*mu
    defectkin=accsq-W2*vsq
    weighted_mean=np.mean(x*v*v)/vsq
    weighted_prediction=(delta+TAU*sigma*W2)/(2*KAPPA)
    dmon=np.linalg.det(M); dtheory=np.exp(-sigma*T)
    return dict(sigma=sigma,g=g,ks=ks,delta=delta,W2=W2,x0=x0,f0=f0,T=T,amplitude=amp,mean_x=mu,mean_x2=x2,
                mean_identity_residual=defectmean,energy_identity_residual=defectkin,
                velocity_weighted_mean=weighted_mean,weighted_mean_predicted=weighted_prediction,
                weighted_mean_residual=weighted_mean-weighted_prediction,
                period_bound=2*np.pi/np.sqrt(W2),period_margin=T-2*np.pi/np.sqrt(W2),
                floquet1_re=np.real(eig[0]),floquet1_im=np.imag(eig[0]),
                floquet2_re=np.real(eig[1]),floquet2_im=np.imag(eig[1]),
                max_nontrivial_floquet=max(abs(eig)),
                monodromy_det=dmon,liouville_det=dtheory,liouville_relative_error=(dmon-dtheory)/dtheory,
                branch_stability=('attracting' if max(abs(eig))<1-1e-7 else 'unstable'))

def first(sigma,offset):
    W2=W0*W0-1/(TAU*TAU)+sigma/TAU
    kh=W0*W0-TAU*sigma*W2
    ks=kh+offset
    eta=np.sign(offset); eps=np.sqrt(abs(offset))
    RH2=-2*TAU*W2*sigma*(4*W2+sigma*sigma)*eta/(KAPPA*KAPPA*(8*W2+sigma*sigma))
    if RH2<=0:return None
    a=eps*np.sqrt(RH2)
    q0=[a,a*(1-sigma*TAU)/(TAU*TAU),2*np.pi/np.sqrt(W2)]
    return correct_cycle(q0,sigma,ks)

def continue_branch(sigma, offsets):
    W2=W0*W0-1/(TAU*TAU)+sigma/TAU
    kh=W0*W0-TAU*sigma*W2
    records=[]; qprev=None; qprevprev=None
    for j,offset in enumerate(offsets):
        ks=kh+offset
        if qprev is None:
            q=first(sigma,offset)
        else:
            qstart=qprev.copy()
            if qprevprev is not None:
                qstart=qprev+(qprev-qprevprev)*((offset-offsets[j-1])/(offsets[j-1]-offsets[j-2]))
            q=correct_cycle(qstart,sigma,ks)
            if q is None: q=correct_cycle(qprev,sigma,ks)
        if q is None:
            print('Continuation stopped:',sigma,ks,offset,'step',j,flush=True)
            break
        r=characterize(q,sigma,ks)
        # Verify expected nontrivial cycle, check identities at numerical quadrature precision
        if r['amplitude']<1e-6 or abs(r['mean_identity_residual'])>2e-5 or abs(r['energy_identity_residual'])>2e-5:
            print('Rejected suspicious cycle',r,flush=True);break
        records.append(r)
        qprevprev=qprev
        qprev=q
    return records

def figure_branches(df,out):
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.2))
    for sig,group in df.groupby('sigma',sort=True):
        group=group.sort_values('ks')
        typ=group.branch_stability.iloc[0]
        color=plt.cm.coolwarm((sig+0.5)/1.35)
        ax=axes[0]
        ax.plot(group.ks,group.amplitude,'-',marker='o',ms=2.8,lw=1.4,color=color,label=fr'$\sigma={sig:.2f}$ ({typ})')
        ax=axes[1]
        ax.plot(group.ks,group['T']/group.period_bound-1,'-',marker='o',ms=2.8,lw=1.4,color=color,label=fr'$\sigma={sig:.2f}$')
    axes[0].set(xlabel=r'$k_s$',ylabel=r'Half peak-to-peak displacement amplitude',title='Nonlinear periodic-orbit branches')
    axes[1].set(xlabel=r'$k_s$',ylabel=r'$T/(2\pi/W)-1$',title='Strict global period bound: excess')
    for ax in axes:ax.grid(alpha=.24)
    axes[0].legend(fontsize=7.5,loc='best')
    fig.tight_layout();fig.savefig(out/'continuation_branches.png',dpi=210);plt.close(fig)

def figure_bifurcation(df,out):
    sigma=np.linspace(-0.54,1.05,360)
    W2=W0**2-1/TAU**2+sigma/TAU
    K0=W0**2-TAU*sigma*W2
    K1=W0**2+TAU*sigma*W2
    fig,ax=plt.subplots(figsize=(8.2,5.0))
    ax.plot(sigma,K0,color='#2d587e',label=r'Hopf at $E_0$')
    ax.plot(sigma,K1,color='#904949',label=r'Hopf at $E_1$')
    ax.axhline(W0**2,lw=1,ls=':',color='#515151',label='equilibrium collision; no periodic orbits')
    # draw verified numeric cycles as small samples at actual parameters; no uncomputed diagram filled in
    for typ,mk,edge in [('attracting','o','#2d6b44'),('unstable','x','#af481e')]:
        g=df[df.branch_stability==typ]
        ax.scatter(g.sigma,g.ks,marker=mk,s=21,color=edge,alpha=.7,label='computed '+typ+' periodic orbit')
    ax.plot([0],[W0**2],marker='*',markersize=12,color='#333333',label='zero--Hopf point')
    ax.set(xlabel=r'$\sigma=d-g+1/\tau$',ylabel=r'$k_s$',title='Spectral Hopf curves and computed periodic cycles',xlim=(-.54,1.05),ylim=(0.35,6.1))
    ax.legend(loc='upper right',fontsize=7.2);ax.grid(alpha=.20)
    fig.tight_layout();fig.savefig(out/'continuation_atlas.png',dpi=210);plt.close(fig)

def mirror_check(r):
    a=(r['ks']-W0*W0)/KAPPA
    # same orbit in translated coordinates solves mirrored k_s equations.
    return dict(sigma=r['sigma'],original_ks=r['ks'],reflected_ks=2*W0**2-r['ks'],shift_x=a,
                translated_mean=r['mean_x']-a,reflected_period=r['T'],reflected_amplitude=r['amplitude'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--outdir',default='.');args=ap.parse_args();out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    branches=[]
    for sig,max_offset in [(0.25,-.15),(0.5,-.45),(0.8,-.38),(-.3,.22),(-.45,.25)]:
        small=.0008 if sig>0 else .0006
        n=34
        offsets=np.sign(max_offset)*np.geomspace(small,abs(max_offset),n)
        result=continue_branch(sig,offsets)
        branches.extend(result)
        if result:print('sigma',sig,'n',len(result),'last',result[-1]['ks'],'amp',result[-1]['amplitude'],'Floquet',result[-1]['max_nontrivial_floquet'],flush=True)
    df=pd.DataFrame(branches)
    if len(df)==0:raise RuntimeError('No converged orbits')
    df.to_csv(out/'continuation_results.csv',index=False,float_format='%.12g')
    figure_branches(df,out);figure_bifurcation(df,out)
    mir=[mirror_check(x) for x in branches]
    pd.DataFrame(mir).to_csv(out/'equilibrium_branch_reflection.csv',index=False,float_format='%.12g')
    mirrored=[]
    for sig in [-.45,-.3,.25,.5,.8]:
        for rr in (df[df.sigma==sig].iloc[[8,-1]]).to_dict('records'):
            a=rr['delta']/KAPPA
            q=np.array([rr['x0']-a,rr['f0']-W0**2*a,rr['T']])
            ks_new=2*W0**2-rr['ks']
            refined=correct_cycle(q,sig,ks_new)
            if refined is None: raise RuntimeError('Mirrored orbit failed direct shooting')
            rnew=characterize(refined,sig,ks_new)
            mirrored.append(dict(sigma=sig,original_ks=rr['ks'],reflected_ks=ks_new,amplitude_difference=rnew['amplitude']-rr['amplitude'],period_difference=rnew['T']-rr['T'],floquet_modulus_difference=rnew['max_nontrivial_floquet']-rr['max_nontrivial_floquet']))
    md=pd.DataFrame(mirrored); md.to_csv(out/'mirrored_shooting_verification.csv',index=False,float_format='%.12g')
    print('Mirror exactness',md.abs().max(numeric_only=True).to_dict(),flush=True)
    print('N cycles',len(df),'Max mean residual',abs(df.mean_identity_residual).max(),'Max energy residual',abs(df.energy_identity_residual).max(),
          'Max weighted residual',abs(df.weighted_mean_residual).max(),
          'Max Liouville rel error',abs(df.liouville_relative_error).max(), 'Min T-bound',df.period_margin.min(),flush=True)
    print('f stability\n',df.groupby(['sigma','branch_stability']).size().to_string(),flush=True)
