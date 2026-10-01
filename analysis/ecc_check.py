import numpy as np
from scipy.optimize import minimize
exec(open('ecc.py').read().split('res={}')[0])
df=S; t=df.t.values; y=df.y.values; e=df.e.values; li=np.array([LABS.index(l) for l in df.lab])
def nll(x):
    P,Tc,K,sec,ses=x[:5]; off=x[5:9]; lj=x[9:13]
    if sec**2+ses**2>=0.95 or K<0: return 1e10
    m=kep_rv(t,P,Tc,K,sec,ses)+off[li]; s2=e**2+np.exp(lj)[li]**2
    return 0.5*np.sum((y-m)**2/s2+np.log(s2))
best=None
for e0 in [0.0,0.5,0.8,0.9]:
    for w0 in np.linspace(0,2*np.pi,8,endpoint=False):
        x0=np.r_[4.26837,2456298.34,5.5 if e0<0.5 else 25,np.sqrt(e0)*np.cos(w0),np.sqrt(e0)*np.sin(w0),np.zeros(4),np.log([5,5,4,4])]
        r=minimize(nll,x0,method='Nelder-Mead',options=dict(maxiter=20000,maxfev=20000,xatol=1e-7,fatol=1e-7))
        if best is None or r.fun<best.fun: best=r
x=best.x; ecc=x[3]**2+x[4]**2
m=kep_rv(t,*x[:5]); i9=np.argmin(abs(t-2454922.53))
print('best nll %.2f  e=%.3f K=%.2f  model at night %.1f (obs-offset %.1f); max model %.1f at BJD %.2f; jitters %s'%(best.fun,ecc,x[2],m[i9],y[i9]-x[5+li[i9]],m.max(),t[np.argmax(m)],np.round(np.exp(x[9:13]),2)))
# circular comparison
xc=best.x.copy()
def nllc(z): return nll(np.r_[z[:3],0,0,z[3:]])
rc=minimize(nllc,np.r_[4.26837,2456298.34,5.5,np.zeros(4),np.log([5,5,4,4])],method='Nelder-Mead',options=dict(maxiter=20000,maxfev=20000))
print('circular nll %.2f -> 2*dlnL(ecc-circ) = %.1f'%(rc.fun,2*(rc.fun-best.fun)))
