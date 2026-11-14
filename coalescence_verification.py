"""Check the center-manifold coefficient and local transcritical stability exchange.
Run with python coalescence_verification.py; writes coalescence_verification.csv.
This is an independent numerical check, not a proof of the proposition.
"""
import csv
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

omega, tau, kappa, d, g = 2.0, 1.0, 1.0, .5, .2
b=d-g; sigma=b+1/tau; W2=omega**2+b/tau
assert sigma>0 and W2>0
coefficient=-kappa/(tau*W2)
# Symbolic coefficient identity of the quadratic center manifold jet.
a,c,D,x=sp.symbols('a c D x')
residual=sp.expand(-sp.Rational(43,10)*(a*D*x+c*x**2) + D*x - x**2)
sol=sp.solve([sp.expand(residual).coeff(D*x),sp.expand(residual).coeff(x,2)],(a,c))
assert sol[a]==sp.Rational(10,43) and sol[c]==-sp.Rational(10,43)

def rhs(t,y,delta):
    x,v,f=y
    return (v, -omega**2*x-b*v+f,( (omega**2+delta)*x-kappa*x*x-f )/tau)

rows=[]
for delta in [-.02, 0., .02]:
    # Equilibrium eigenvalues; for delta<0 E0 stable; for delta>0 E1 stable.
    for label,xe in [('E0',0.),('E1',delta/kappa)]:
        J=np.array([[0.,1.,0.],[-omega**2,-b,1.],[(omega**2+delta-2*kappa*xe)/tau,0.,-1./tau]])
        eig=np.linalg.eigvals(J)
        rows.append({'test':'equilibrium','delta':delta,'side':label,'t':0,
                     'x_actual':xe,'x_reduced':np.nan,'max_real_eig':max(eig.real),'relative_error':np.nan})

for x0 in [0.02,-0.02]:
    v0=coefficient*x0*x0
    f0=omega**2*x0+b*v0
    out=solve_ivp(lambda t,y: rhs(t,y,0.),(0,80),(x0,v0,f0),
                  t_eval=[0.,20.,40.,80.],rtol=2e-12,atol=1e-13,method='DOP853')
    assert out.success
    for j,t in enumerate(out.t):
        approximate=x0/(1-coefficient*x0*t)
        actual=out.y[0,j]
        rows.append({'test':'center_manifold','delta':0.,'side':'attracting' if x0>0 else 'repelling',
                    't':t,'x_actual':actual,'x_reduced':approximate,
                    'max_real_eig':np.nan,'relative_error':abs(actual-approximate)/abs(approximate)})

with open(Path(__file__).with_name('coalescence_verification.csv'),'w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('sigma',sigma,'W2',W2,'center coefficient',coefficient)
for r in rows:
    if r['test']=='center_manifold' and r['t']==80:
        print(r['side'],'x_actual',r['x_actual'],'approx',r['x_reduced'],'relative_error',r['relative_error'])
print('Results written:',len(rows),'rows')
