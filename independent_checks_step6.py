"""Small exact symbolic audit for Revision 6; not a complete proof certification.
Requires sympy. Run: python independent_checks_step6.py
"""
import sympy as S
lamb=S.symbols('lam')
w0,tau,d,g,ks,kappa,delta,x,v,f=S.symbols('w0 tau d g ks kappa delta x v f', real=True)
sigma=d-g+1/tau
W2=w0**2+(d-g)/tau
J0=S.Matrix([[0,1,0],[-w0**2,g-d,1],[ks/tau,0,-1/tau]])
poly=S.expand(J0.charpoly(lamb).as_expr())
expected=lamb**3+sigma*lamb**2+W2*lamb+(w0**2-ks)/tau
assert S.simplify(poly-expected)==0, 'Origin characteristic polynomial mismatch'
ks1=2*w0**2-ks
J1=J0.subs(ks,ks1)
assert S.simplify(J1.charpoly(lamb).as_expr()-(lamb**3+sigma*lamb**2+W2*lamb+(ks-w0**2)/tau))==0, 'Reflected characteristic polynomial mismatch'
X=S.Matrix([v,-w0**2*x-(d-g)*v+f,(ks*x-f-kappa*x*x)/tau])
xstar=(ks-w0**2)/kappa
trans=S.Matrix([xstar,0,w0**2*xstar])
Y=S.Matrix([x,v,f]); XT=S.simplify(X.subs(dict(zip(Y,Y+trans)), simultaneous=True))
reflect=S.simplify(X.subs(ks,ks1))
assert all(S.simplify(a-b)==0 for a,b in zip(XT,reflect)), 'Affine reflection mismatch'
assert S.simplify(S.diff(X[0],x)+S.diff(X[1],v)+S.diff(X[2],f)+sigma)==0, 'Divergence mismatch'
# Check cubic at a standard Hopf point and the real-part crossing derivative.
Ssigma,Om=S.symbols('sig Om', positive=True)
kh=w0**2-tau*Ssigma*(w0**2-1/tau**2+Ssigma/tau)
poly_h=expected.subs({g:d+1/tau-Ssigma,ks:kh})
Om2=w0**2-1/tau**2+Ssigma/tau
assert S.simplify(poly_h-(lamb+Ssigma)*(lamb*lamb+Om2))==0, 'Hopf factorization mismatch'
# Evaluate transversality at a representative physically admissible Hopf point,
# including a case with w0*tau <= 1 (outside the zero-Hopf regime).
for W0, Tau, Sig in [(S.Rational(2), S.Integer(1), S.Rational(1,2)),
                      (S.Rational(3,5), S.Integer(1), S.Integer(1))]:
    om2=(W0**2-Tau**(-2)+Sig/Tau)
    assert om2>0
    p_lam=S.diff(poly,lamb).subs({w0:W0,tau:Tau,d:S.Rational(1,5),g:S.Rational(1,5)+1/Tau-Sig,
      ks:(W0**2-Tau*Sig*om2),lamb:S.I*S.sqrt(om2)})
    deriv=S.simplify(1/(Tau*p_lam))
    target=-1/(2*Tau*(om2+Sig**2))
    assert S.simplify(S.re(deriv)-target)==0, 'Hopf transversality mismatch'
# Chart boundary dynamics for compactification; check all three positive charts.
u,q,ww,a,b,c,h,k=S.symbols('u q ww a b c h k', real=True)
orig=S.Matrix([v,-a*x-b*v+f,c*x-h*f-k*x*x])
state=(x,v,f)
U1={x:1/ww,v:u/ww,f:q/ww}
U2={x:u/ww,v:1/ww,f:q/ww}
U3={x:u/ww,v:q/ww,f:1/ww}
chart_coordinates=[(v/x,f/x,1/x), (x/v,f/v,1/v), (x/f,v/f,1/f)]
expected_borders=[(0,-k,0),(0,-k*u*u,0),(k*u**3,k*u*u*q,0)]
for i,(sub,coord,exp) in enumerate(zip((U1,U2,U3),chart_coordinates,expected_borders),start=1):
    new=[S.factor((ww*sum(S.diff(cvar,s)*orig[j] for j,s in enumerate(state))).subs(sub,simultaneous=True)) for cvar in coord]
    border=[S.simplify(expr.subs(ww,0)) for expr in new]
    assert all(S.simplify(A-B)==0 for A,B in zip(border,exp)), f'Chart U{i} boundary mismatch: {border}'
# The zero-Hopf averaged Jacobian trace and determinant have the asserted forms.
r,gg,kk,zz,TT,kap,ff=S.symbols('r gg kk zz TT kap ff', real=True, nonzero=True)
F1=S.pi*r/zz*(gg-kk/(zz**2*TT)+2*kap*ff/(zz**2*TT*w0**2))
F2=S.pi/(zz**3*TT)*(2*kk*ff-kap*w0**2*r*r-2*kap*ff*ff/w0**2)
r0sq=(kk*kk-gg*gg*zz**4*TT**2)/(2*kap*kap)
f0=w0**2/(2*kap)*(kk-gg*zz**2*TT)
D=S.Matrix([F1,F2]).jacobian([r,ff])
tr=S.simplify(S.trace(D).subs(ff,f0));det=S.factor(D.det().subs({ff:f0,r*r:r0sq}))
assert S.simplify(tr-2*S.pi*gg/zz)==0,'Averaged Jacobian trace mismatch'
assert S.simplify(det-2*S.pi**2*(kk*kk-gg*gg*zz**4*TT**2)/(zz**6*TT**2))==0,'Averaged Jacobian determinant mismatch'
print('PASS: 9 symbolic groups: characteristic polynomials, branch reflection, divergence, Hopf factorization/transversality, 3 boundary charts, averaged Jacobian trace/determinant.')
print('LIMIT: These checks do not independently certify higher-order averaging, analytic continuation, or nonlinear experimental modeling.')
