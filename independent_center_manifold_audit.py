"""Independent fifth-order Hopf center-manifold homological audit.
Uses vector-field Taylor tensors and a 3D complex center manifold rather
than scalar jerk Poincare-Lindstedt/harmonic balance or orbit fitting.
"""
import numpy as np
from math import sqrt
from collections import defaultdict
W=sqrt(3.5); ks=2.25; b=-0.5; sig=0.5
L2=290871/57856
LAM=sqrt(L2)
a={2:-1.,3:-ks**3/(3*L2),4:ks**2/L2,5:-ks/L2+2*ks**5/(15*L2**2)}
J=np.array([[0.,1.,0.],[-4.,-b,1.],[ks,0.,-1.]],dtype=complex)
q=np.array([.5, .5j*W, .5*(-W**2+b*1j*W+4)], dtype=complex)
assert np.max(np.abs(J@q-1j*W*q))<1e-14
vecs, ev = np.linalg.eig(J)
# left projection p^H with p^H q=1 and p^H conj(q)=0 and p^H stable mode=0
stable=ev[:,np.argmin(abs(vecs+.5))]
Q=np.column_stack((q,q.conj(),stable))
pstar=np.linalg.inv(Q)[0,:]  # row, not conjugation of input
assert abs(pstar@q-1)<1e-13
assert abs(pstar@q.conj())<1e-13
# H[(j,k)] is vector coefficient for w^j z^k
H={(1,0):q,(0,1):q.conj()}
G={}
def mul(u,v,nmax=5):
    r=defaultdict(complex)
    for (j,k),v1 in u.items():
        for (l,m),v2 in v.items():
            if j+k+l+m <=nmax:r[(j+l,k+m)]+=v1*v2
    return dict(r)
def nonlin_degree(n):
    x={idx:vec[0] for idx,vec in H.items()}
    power={(0,0):1.+0j}; vals=defaultdict(complex)
    for degree in range(1,6):
        power=mul(power,x)
        if degree in a:
            for key,val in power.items():
                if sum(key)==n: vals[key]+=a[degree]*val
    return {key:np.array([0.,0.,z],dtype=complex) for key,z in vals.items()}
def nonlinear_chain(n):
    res=defaultdict(lambda:np.zeros(3,dtype=complex))
    if not G:return res
    for (j,k),hv in H.items():
        for (l,m),gv in G.items():
            # wdot contains G_(l,m) w^l z^m, zdot contains conj(G_(l,m)) w^m z^l
            if j and j+k+l+m-1==n:res[(j-1+l,k+m)]+=j*hv*gv
            if k and j+k+l+m-1==n:res[(j+m,k-1+l)]+=k*hv*gv.conjugate()
    return res
for n in range(2,6):
    NL=nonlin_degree(n); CH=nonlinear_chain(n)
    for k in range(n+1):
        j=n-k
        if j < k:continue  # enforce conjugate coefficients
        rhs=NL.get((j,k),np.zeros(3,dtype=complex))-CH.get((j,k),np.zeros(3,dtype=complex))
        M=(1j*W*(j-k))*np.eye(3)-J
        if j-k==1:
            K=np.zeros((4,4),dtype=complex)
            K[:3,:3]=M;K[:3,3]=q;K[3,:3]=pstar
            ans=np.linalg.solve(K,np.r_[rhs,0.+0.j]);h=ans[:3];g=ans[3]
            G[(j,k)]=g
            print('degree',n,'resonance', (j,k), 'g=',g,'residual=',np.max(abs(M@h+q*g-rhs)))
        else:
            h=np.linalg.solve(M,rhs)
        H[(j,k)]=h
        if j!=k:H[(k,j)]=h.conj()
# two equivalent normal form amplitude conventions may differ by nonlinear coordinate gauge.
print('cubic reduced coefficient:',G[(2,1)])
print('quintic reduced coefficient:',G[(3,2)])
print('quintic real coefficient:',G[(3,2)].real)
target=265240779872/1089070956855
assert abs(G[(2,1)].real)<1e-13
assert abs(G[(3,2)].real-target)<1e-12
print('P-L target:',target)
from pathlib import Path
import json
Path(__file__).with_name('center_manifold_audit_results.json').write_text(json.dumps({
    'g21_real': float(G[(2,1)].real), 'g21_imag':float(G[(2,1)].imag),
    'g32_real': float(G[(3,2)].real), 'g32_imag':float(G[(3,2)].imag),
    'Poincare_Lindstedt_gamma':target,
    'absolute_difference':float(abs(G[(3,2)].real-target)),
    'linear_eigenvalue_frequency':W,
    'amplitude_normalization':'q_x=1/2, x leading term A cos(Omega t)'
}, indent=2)+'\n')
