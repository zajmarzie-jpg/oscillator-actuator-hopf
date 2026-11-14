"""Reproducible Step 3 analysis: pseudo-arclength shooting and force-command saturation.

This code is numerical evidence, NOT verified/exhaustive global continuation.
For saturation, the *entire commanded actuator force* is bounded by Fmax.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W0=2.; TAU=1.; KAPPA=1.; D=.2
RTOL=6e-11; ATOL=8e-13

def flow(t,y,sigma,ks,fmax=np.inf):
    x,v,f=y[:3]
    g=D+1./TAU-sigma
    command=ks*x-KAPPA*x*x
    if np.isfinite(fmax):
        tnh=np.tanh(command/fmax)
        target=fmax*tnh
        dtarget=(ks-2*KAPPA*x)*(1.-tnh*tnh)
    else:
        target=command
        dtarget=ks-2*KAPPA*x
    Fy=np.array([v,-W0**2*x+(g-D)*v+f,(target-f)/TAU])
    if len(y)==3:return Fy
    J=np.array([[0.,1.,0.],[-W0**2,g-D,1.],[dtarget/TAU,0.,-1./TAU]])
    M=y[3:12].reshape(3,3)
    output=np.r_[Fy,(J@M).ravel()]
    if len(y)>12:
        Sk=y[12:15]
        dfdks=x if not np.isfinite(fmax) else x*(1.-tnh*tnh)
        output=np.r_[output,J@Sk+[0,0,dfdks/TAU]]
    return output

def shoot(q,sigma,fmax=np.inf,sens=False,dense=False):
    x0,f0,T,ks=q
    y0=np.r_[x0,0.,f0,np.eye(3).ravel()]
    if sens:y0=np.r_[y0,np.zeros(3)]
    return solve_ivp(flow,(0.,T),y0,args=(sigma,ks,fmax),method='DOP853',rtol=RTOL,atol=ATOL,dense_output=dense)

def residual_jac(q,sigma,fmax=np.inf,sens=False):
    if not np.all(np.isfinite(q)) or q[2]<1.0 or q[2]>25 or np.max(np.abs(q))>2e2:
        raise ValueError('invalid shooting state')
    sol=shoot(q,sigma,fmax,sens=sens)
    if not sol.success:raise RuntimeError(sol.message)
    ye=sol.y[:3,-1];M=sol.y[3:12,-1].reshape(3,3)
    r=ye-np.array([q[0],0.,q[1]])
    J=np.column_stack([M[:,0]-[1,0,0],M[:,2]-[0,0,1],flow(q[2],ye,sigma,q[3],fmax)])
    if sens:J=np.column_stack([J,sol.y[12:15,-1]])
    return r,J

def characterize(q,sigma,fmax=np.inf):
    sol=shoot(q,sigma,fmax,dense=True)
    t=np.linspace(0,q[2],900,endpoint=False)
    traj=sol.sol(t)[:3]
    xx,vv,ff=traj
    cmd=q[3]*xx-KAPPA*xx*xx
    if np.isfinite(fmax):
        hyper=np.tanh(cmd/fmax)
        F=fmax*hyper
        Fprime=(q[3]-2*KAPPA*xx)*(1-hyper*hyper)
    else:
        F=cmd
        Fprime=q[3]-2*KAPPA*xx
    acc=-W0**2*xx+(D+1./TAU-sigma-D)*vv+ff
    W2=W0**2+(-1./TAU+sigma)/TAU
    err_mean=np.mean(F)-W0**2*np.mean(xx)
    err_energy=np.mean(acc**2)-W2*np.mean(vv**2)
    err_slope=np.mean(Fprime*vv**2)/np.mean(vv**2)-(W0**2-TAU*sigma*W2)
    M=sol.y[3:12,-1].reshape(3,3)
    eig=np.linalg.eigvals(M)
    phase=np.argmin(np.abs(eig-1))
    others=np.delete(eig,phase)
    amp=(xx.max()-xx.min())/2
    return dict(sigma=sigma,ks=q[3],x0=q[0],f0=q[1],T=q[2],amplitude=amp,mean_x=np.mean(xx),
                lambda1_re=others[0].real,lambda1_im=others[0].imag,lambda2_re=others[1].real,lambda2_im=others[1].imag,
                floquet_modulus=max(abs(others)),phase_multiplier_error=abs(eig[phase]-1),
                shooting_residual=np.max(abs(sol.y[:3,-1]-[q[0],0,q[1]])),
                mean_force_residual=err_mean,energy_residual=err_energy,slope_residual=err_slope,
                liouville_relative_error=np.linalg.det(M)/np.exp(-sigma*q[2])-1,
                Fmax=fmax, type='attracting' if max(abs(others))<1.-1e-7 else 'unstable')

def pseudo_step(qprev,qcur,sigma,step):
    tangent=qcur-qprev
    tangent=tangent/np.linalg.norm(tangent)
    pred=qcur+step*tangent
    def f(z):
        r,_=residual_jac(z,sigma,sens=True)
        return np.r_[r,np.dot(z-pred,tangent)]
    def jac(z):
        _,J=residual_jac(z,sigma,sens=True)
        return np.vstack([J,tangent])
    try:
        opt=root(f,pred,jac=jac,tol=4e-10)
        if not np.all(np.isfinite(opt.x)) or np.max(abs(f(opt.x)))>2e-7:return None
        if abs(opt.x[0])<1e-5 or flow(0,np.r_[opt.x[0],0,opt.x[1]],sigma,opt.x[3])[1]>=0:return None
        return opt.x
    except (ValueError, RuntimeError, OverflowError):return None

def continue_arc(df,sigma,nsteps=23):
    g=df[np.isclose(df.sigma,sigma)].sort_values('ks')
    kh=W0*W0-sigma*(W0*W0-1.+sigma)
    # away from Hopf, choose the two samples farthest from ks_H
    g=g.assign(distance=(g.ks-kh).abs()).sort_values('distance')
    prev,cur=g.iloc[-2],g.iloc[-1]
    a=np.array([prev.x0,prev.f0,prev["T"],prev.ks],float)
    b=np.array([cur.x0,cur.f0,cur["T"],cur.ks],float)
    records=[]
    baseline=characterize(b,sigma); baseline.update(step=0,arclength_step=0.,newton_halvings=0)
    records.append(baseline)
    step=max(np.linalg.norm(b-a),.015)
    step=min(step,.045)
    for j in range(1,nsteps+1):
        trial=None
        halvings=0
        local=step
        while trial is None and halvings<=5:
            trial=pseudo_step(a,b,sigma,local)
            if trial is None:
                local*=.5;halvings+=1
        if trial is None:break
        c=characterize(trial,sigma)
        if c['amplitude']<1e-5 or c['shooting_residual']>1e-7:break
        c.update(step=j,arclength_step=local,newton_halvings=halvings)
        records.append(c)
        a,b=b,trial
        step=min(local*1.06,.05)
    return records

def saturation_cases(df, sigmas=(-.3,.5), cases=(10.,25.,100.)):
    rows=[]
    for sigma in sigmas:
        g=df[np.isclose(df.sigma,sigma)].sort_values('ks')
        kh=W0*W0-sigma*(W0*W0-1.+sigma)
        g=g.assign(distance=abs(g.ks-kh)).sort_values('distance')
        # moderate and largest sampled amplitude
        for marker,ref in [('moderate',g.iloc[18]),('end',g.iloc[-1])]:
            refq=np.array([ref.x0,ref.f0,ref["T"],ref.ks],float)
            base=characterize(refq,sigma)
            base.update(case=marker,relative_amplitude_shift=0.,relative_period_shift=0.,
                        ks_fixed=ref.ks,converged=True,reference_amplitude=base['amplitude'])
            rows.append(base)
            init=refq.copy()
            for fm in sorted(cases,reverse=True):
                def resid3(u):
                    return residual_jac(np.r_[u,ref.ks],sigma,fm)[0]
                def jac3(u):
                    return residual_jac(np.r_[u,ref.ks],sigma,fm)[1][:,:3]
                try:
                    opt=root(resid3,init[:3],jac=jac3,tol=4e-10)
                    qq=np.r_[opt.x,ref.ks]
                    if np.max(abs(resid3(opt.x)))>1e-7 or abs(qq[0])<1e-5:
                        raise ValueError('not a corrected nonconstant orbit')
                    re=characterize(qq,sigma,fm)
                    if re['amplitude']<1e-5:raise ValueError('collapsed orbit')
                    re.update(case=marker,reference_amplitude=base['amplitude'],converged=True,
                              relative_amplitude_shift=(re['amplitude']/base['amplitude']-1),
                              relative_period_shift=(re['T']/base['T']-1),ks_fixed=ref.ks)
                    init=qq
                except Exception as ex:
                    re=dict(sigma=sigma,ks=ref.ks,Fmax=fm,case=marker,converged=False,failure=str(ex))
                rows.append(re)
    return rows

def make_figure(arc, sat, out):
    fig,axs=plt.subplots(1,2,figsize=(10.7,4.3))
    for sig,grp in arc.groupby('sigma'):
        axs[0].plot(grp.ks,grp.amplitude,'o-',markersize=3,label=f'sigma={sig:.2f}')
        axs[1].plot(grp.ks,grp.floquet_modulus,'o-',markersize=3,label=f'sigma={sig:.2f}')
    axs[0].set(xlabel=r'$k_s$',ylabel='Half peak-to-peak amplitude',title='Pseudo-arclength branches')
    axs[1].set(xlabel=r'$k_s$',ylabel='Maximum nontrivial Floquet modulus',title='Orbital stability along branches')
    axs[1].axhline(1.,linestyle='--',linewidth=1,color='gray')
    for ax in axs:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'pseudo_arclength_branches.png',dpi=195);plt.close(fig)
    sat=sat[sat.converged==True].copy()
    if len(sat)>0:
        fig,axs=plt.subplots(1,2,figsize=(10.7,4.3))
        for (sig,case),grp in sat.groupby(['sigma','case']):
            grp=grp.sort_values('Fmax',ascending=False)
            finite=grp[np.isfinite(grp.Fmax)]
            axs[0].plot(finite.Fmax,100*finite.relative_amplitude_shift,'o-',label=f'sigma={sig:.2f}, {case}')
            axs[1].plot(finite.Fmax,finite.floquet_modulus,'o-',label=f'sigma={sig:.2f}, {case}')
        axs[0].set(xlabel='Command saturation level $F_{max}$',ylabel='Amplitude change (%)',title='Bounded-force sensitivity')
        axs[1].set(xlabel='Command saturation level $F_{max}$',ylabel='Nontrivial Floquet modulus',title='Stability under saturation')
        axs[1].axhline(1,ls='--',lw=1,color='gray')
        for ax in axs:ax.grid(alpha=.25);ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/'saturation_robustness.png',dpi=195);plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--outdir',default='.');args=ap.parse_args()
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(out/'continuation_results.csv')
    records=[]
    for sig in [0.5,-0.3]:
        branch=continue_arc(df,sig)
        records.extend(branch)
        print('Arc sigma:',sig,'converged:',len(branch),'end ks:',branch[-1]['ks'],'max modulus:',branch[-1]['floquet_modulus'],flush=True)
    arc=pd.DataFrame(records);arc.to_csv(out/'pseudo_arclength_results.csv',index=False,float_format='%.13g')
    sat=pd.DataFrame(saturation_cases(df));sat.to_csv(out/'saturation_results.csv',index=False,float_format='%.13g')
    print('Saturation:',sat[['sigma','case','Fmax','converged','amplitude','floquet_modulus','relative_amplitude_shift']].to_string(index=False),flush=True)
    make_figure(arc,sat,out)
    print('Arc residual max:',arc.shooting_residual.max(),'max tangent halves:',arc.newton_halvings.max(),flush=True)
    for label,tab in [('arc',arc),('saturation',sat[sat.converged==True])]:
        print(label,'exact constraints max abs:',{k:float(tab[k].abs().max()) for k in ['mean_force_residual','energy_residual','slope_residual','liouville_relative_error']},flush=True)
