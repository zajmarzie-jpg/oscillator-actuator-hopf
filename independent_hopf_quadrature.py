"""Independent high-resolution periodic quadrature of second-order Hopf return coefficient.
Checks the explicit secular-displacement integrand from the transformed vector field,
not a regression against previously generated periodic-orbit data.
"""
import csv
import numpy as np
from pathlib import Path
from scipy.integrate import simpson

HERE=Path(__file__).resolve().parent
rows=[]
for tau, omega0, kappa, sigma, eta2, R in [
    (1.,2.,1.,.4,-.7,1.2),
    (1.,2.,-.7,1.1,-1.3,.95),
    (1.,2.,1.4,-.55,.6,1.1),
    (.7,1.1,1.8,.9,-.4,.85),
    (.75,2.,-.45,-.3,.2,1.7),
]:
    Om2=omega0**2-1/tau**2+sigma/tau
    assert Om2>0 and sigma!=0
    Om=np.sqrt(Om2); D=Om2+sigma**2
    theta=np.linspace(0,2*np.pi,8193,endpoint=True)
    c=np.cos(theta); s=np.sin(theta)
    A=(Om*c+sigma*s)/(Om*D)
    B=(Om*s-sigma*c)/(Om*D)
    # Zeroth transverse coordinate W=0; first periodic correction w1 from exact forced ODE.
    r1=(kappa*R**2/(tau*Om**2*D))*(Om*(s-s**3/3)+sigma*(1-c**3)/3)
    w1=(-kappa*R**2/(2*tau*sigma*D)
        -kappa*R**2*sigma/(2*tau*D*(4*Om2+sigma**2))*np.cos(2*theta)
        -kappa*R**2*Om/(tau*D*(4*Om2+sigma**2))*np.sin(2*theta))
    F2first=-(eta2*R*c)/(tau*Om)*A+(kappa**2*R**3*c**4)/(tau**2*Om**2)*A*B
    dR_F1=2*kappa*R*c**2*A/(tau*Om)
    dW_F1=2*kappa*R*c*A/(tau*Om)
    integrand=F2first+dR_F1*r1+dW_F1*w1
    numeric=simpson(integrand, x=theta)
    C=kappa**2*(8*Om2+sigma**2)/(2*tau*Om2*sigma*(4*Om2+sigma**2))
    symbolic=-np.pi*R/(tau*Om*D)*(eta2+C*R**2)
    rel=abs(numeric-symbolic)/max(1,abs(symbolic))
    # The periodic w1 is independently checked against its defining linear equation.
    rhs=-sigma/Om*w1-kappa*R**2*c**2/(tau*Om*D)
    lhs=(kappa*R**2*sigma/(tau*D*(4*Om2+sigma**2))*np.sin(2*theta)
         -2*kappa*R**2*Om/(tau*D*(4*Om2+sigma**2))*np.cos(2*theta))
    wr=abs(lhs-rhs).max()
    rows.append(dict(tau=tau,omega0=omega0,kappa=kappa,sigma=sigma,
        eta2=eta2,R=R,omega=Om,quad_coefficient=numeric,
        paper_coefficient=symbolic,relative_error=rel,transverse_ode_max_residual=wr))
    print(f'sigma={sigma:+.3f}, quad={numeric:+.12g}, paper={symbolic:+.12g}, rel={rel:.2e}; w1 residual={wr:.2e}')
    assert rel < 3e-10 and wr<2e-12
with (HERE/'hopf_independent_audit.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('All 5 direct second-order Hopf quadrature comparisons passed.')
