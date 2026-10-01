import numpy as np
from scipy.optimize import minimize
exec(open('ecc_check.py').read().split('best=None')[0])
tn=2454922.527; P=4.26837
best=None
for ecc in [0.8,0.85,0.9,0.93]:
    for w in np.linspace(0,2*np.pi,24,endpoint=False):
        # periastron time tp = tn ; convert to Tc
        f_c=np.pi/2-w; E_c=2*np.arctan(np.sqrt((1-ecc)/(1+ecc))*np.tan(f_c/2)); M_c=E_c-ecc*np.sin(E_c)
        Tc=tn+P*M_c/(2*np.pi)
        def nll2(z):
            return nll(np.r_[P,Tc,z[0],np.sqrt(ecc)*np.cos(w),np.sqrt(ecc)*np.sin(w),z[1:5],z[5:9]])
        r=minimize(nll2,np.r_[25,np.zeros(4),np.log([5,5,4,4])],method='Nelder-Mead',options=dict(maxiter=8000,maxfev=8000))
        if best is None or r.fun<best[0]: best=(r.fun,ecc,w,Tc,r.x)
print('best high-e nll %.2f e=%.2f K=%.1f jit %s'%(best[0],best[1],best[4][0],np.round(np.exp(best[4][5:9]),2)))
x=best; r=minimize(nll,np.r_[P,x[3],x[4][0],np.sqrt(x[1])*np.cos(x[2]),np.sqrt(x[1])*np.sin(x[2]),x[4][1:]],method='Nelder-Mead',options=dict(maxiter=40000,maxfev=40000,xatol=1e-8,fatol=1e-8))
m=kep_rv(t,*r.x[:5]); i9=np.argmin(abs(t-2454922.53))
print('refined nll %.2f e=%.3f K=%.1f model at night %.1f jit %s'%(r.fun,r.x[3]**2+r.x[4]**2,r.x[2],m[i9],np.round(np.exp(r.x[9:13]),2)))
