"""Independent symbolic checks for Rev.7. Run with Python and SymPy."""
import sympy as S
th=S.symbols('theta', real=True)
Om,T,w0,kap,g1,k1,r,u=S.symbols('Omega tau omega_0 kappa g1 k1 r u', positive=True,real=True)
# Independent construction of O(epsilon) polar dynamics from the coordinate map.
H=r*S.cos(th)+u/w0**2
P=g1*r*S.sin(th)/(Om*T)-k1*H/(Om**2*T)+kap*H**2/(Om**2*T)
Q=g1*r*S.sin(th)
R=-g1*w0**2*r*S.sin(th)/(Om*T)+k1*w0**2*H/(Om**2*T)-kap*w0**2*H**2/(Om**2*T)
F1=(P*S.cos(th)+Q*S.sin(th))/Om
F2=R/Om
result=[S.simplify(S.integrate(S.expand_trig(S.expand(ff)),(th,0,2*S.pi))) for ff in [F1,F2]]
expected1=S.pi*r/Om*(g1-k1/(Om**2*T)+2*kap*u/(Om**2*T*w0**2))
expected2=S.pi/(Om**3*T)*(2*k1*u-kap*w0**2*r**2-2*kap*u**2/w0**2)
assert S.simplify(result[0]-expected1)==0
assert S.simplify(result[1]-expected2)==0
print('PASS: independent integral of both zero-Hopf first averaged components.')
# General existence condition and discriminant at g1=0.
r0sq=k1**2/(2*kap**2)
u0=w0**2*k1/(2*kap)
D=S.Matrix([expected1,expected2]).jacobian([r,u])
D0=D.subs({g1:0,u:u0})
assert S.simplify(S.trace(D0))==0
assert S.simplify(D0.det().subs(r**2,r0sq)-2*S.pi**2*k1**2/(Om**6*T**2))==0
nu=S.sqrt(2)*S.pi*k1/(Om**3*T)
assert S.simplify(S.sqrt(D0.det().subs(r**2,r0sq))-nu)==0
print('PASS: trace zero, positive determinant and elliptic averaged spectrum.')
# Embed exact scalar jerk equation in general quadratic jerk polynomial form.
x,v,z,f,delta,sigma,W2,zz=S.symbols('x v z f delta sigma W2 z', real=True)
jerk=-sigma*z-W2*v+delta*x/T-kap*x*x/T
coeffs=[S.expand(jerk).coeff(q) for q in [x,v,z,x*x]]
assert coeffs==[delta/T,-W2,-sigma,-kap/T]
print('PASS: restricted quadratic-jerk coefficient mapping (a1,a2,a3,a4).')

# Alternative Hopf cubic coefficient via order-A^3 harmonic balance.
sg=S.symbols('sigma',positive=True, real=True)
Om2=S.symbols('Omega_H',positive=True, real=True)
c0=-kap/(2*T*sg*Om2**2)
c2=kap*sg/(6*T*Om2**2*(sg**2+4*Om2**2))
s2=kap/(3*T*Om2*(sg**2+4*Om2**2))
nu=-kap*s2/(2*T*Om2**2)
mu2=S.factor(kap*(2*c0+c2)-2*T*Om2*sg*nu)
assert S.simplify(mu2 + kap**2*(8*Om2**2+sg**2)/(2*T*Om2**2*sg*(4*Om2**2+sg**2)))==0
y2=c0+c2*S.cos(2*th)+s2*S.sin(2*th)
L2=Om2**3*S.diff(y2,th,3)+sg*Om2**2*S.diff(y2,th,2)+Om2**3*S.diff(y2,th)+sg*Om2**2*y2
assert S.simplify(S.trigsimp(S.expand_trig(L2+kap/(2*T)*(1+S.cos(2*th)))))==0
print('PASS: independent harmonic-balance recovery of the ordinary Hopf cubic coefficient.')
