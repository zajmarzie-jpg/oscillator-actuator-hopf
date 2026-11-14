"""Symbolic, independent third-order harmonic-balance coefficient audit.
Recomputes the response of the quadratic forcing at zero and double
frequency and the cubic fundamental; checks it equals the manuscript's
closed-form real-amplitude coefficient and threshold.
"""
import sympy as sp
sig,Om,tau,a,c,ks,kap,Lam=sp.symbols('sig Om tau a c ks kap Lam',real=True,nonzero=True)
L2=-3*Om**2*(sig+2*sp.I*Om)
Lp=-2*Om**2+2*sp.I*sig*Om
alpha=sp.factor(sp.re(1/(tau*Lp)))
beta=sp.factor(sp.re((a**2/tau**2*(1/(sig*Om**2)+1/(2*L2))+3*c/(4*tau))/Lp))
beta_paper=-a**2*(8*Om**2+sig**2)/(4*tau**2*Om**2*sig*(Om**2+sig**2)*(4*Om**2+sig**2))-3*c/(8*tau*(Om**2+sig**2))
assert sp.simplify(beta-beta_paper)==0
assert sp.simplify(alpha+1/(2*tau*(Om**2+sig**2)))==0
beta_sat=sp.factor(beta.subs({a:-kap,c:-ks**3/(3*Lam**2)}))
crit=ks**3*tau*Om**2*sig*(4*Om**2+sig**2)/(2*kap**2*(8*Om**2+sig**2))
assert sp.simplify(beta_sat.subs(Lam**2,crit))==0
print('PASS: independent mean+second-harmonic Hopf coefficient, spectral crossing, and cubic-zero threshold.')
print('Alpha:',alpha)
print('Beta:',beta_sat)
