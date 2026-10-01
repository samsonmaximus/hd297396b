# Profile likelihood of the quasi-periodic model over the rotation period, 10-100 d, for dLW alone and for
# the joint velocity + dLW model; plus a periodic-free control (w -> large) to measure what the periodic
# term adds. Offsets profiled by GLS, jitters and hyperparameters optimised at each fixed period.
import numpy as np, pandas as pd, json
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
G=pd.read_csv('frozen104.csv')
g=G[np.abs(G.t-2454922.53)>0.2].reset_index(drop=True)
t=g.t.values-2455000; r=t[:,None]-t[None,:]; li=g.li.values; Lm=(li[:,None]==np.arange(4)[None,:]).astype(float)
lw,elw,rv,erv=g.lw.values,g.elw.values,g.rv.values,g.erv.values
def gll(y,e,jit,K0):
    K=K0.copy(); K[np.diag_indices_from(K)]+=e**2+jit[li]**2
    c=cho_factor(K,lower=True,check_finite=False)
    CiD=cho_solve(c,Lm,check_finite=False); Ciy=cho_solve(c,y,check_finite=False)
    p=np.linalg.solve(Lm.T@CiD,Lm.T@Ciy); rr=y-Lm@p
    return -0.5*rr@cho_solve(c,rr,check_finite=False)-np.sum(np.log(np.diag(c[0])))
def kqp(lam,P,w): return np.exp(-r*r/(2*lam*lam)-np.sin(np.pi*r/P)**2/(2*w*w))
def prof(P,joint):
    # x: ln jl(4), ln Al, ln lam, ln w [, ln jr(4), ln Ar]
    def nll(x):
        k=kqp(np.exp(x[5]),P,np.exp(x[6]))
        v=-gll(lw,elw,np.exp(x[0:4]),np.exp(2*x[4])*k)
        if joint: v-=gll(rv,erv,np.exp(x[7:11]),np.exp(2*x[11])*k)
        return v
    bnd=[(np.log(0.01),np.log(50))]*4+[(np.log(0.1),np.log(500)),(np.log(10),np.log(1000)),(np.log(0.1),np.log(2))]
    if joint: bnd+= [(np.log(0.01),np.log(50))]*4+[(np.log(0.01),np.log(50))]
    best=None
    for w0 in [0.3,0.7,1.5]:
        x0=np.r_[np.log([3,1,5,1]),np.log(15),np.log(300),np.log(w0)]
        if joint: x0=np.r_[x0,np.log([4,5,4,4]),np.log(1)]
        rr=minimize(nll,x0,method='L-BFGS-B',bounds=bnd)
        if best is None or rr.fun<best.fun: best=rr
    return -best.fun, np.exp(best.x[6])
Ps=np.exp(np.linspace(np.log(10),np.log(100),181)); out={}
for joint in [False]:
    ll=[];ww=[]
    for P in Ps:
        a,b=prof(P,joint); ll.append(a); ww.append(b)
    ll=np.array(ll); i=np.argmax(ll)
    # no-periodic control: w at its upper bound 2 is nearly aperiodic; use a squared-exponential only
    def nll_se(x):
        k=np.exp(-r*r/(2*np.exp(2*x[5])))
        return -gll(lw,elw,np.exp(x[0:4]),np.exp(2*x[4])*k)
    se=minimize(nll_se,np.r_[np.log([3,1,5,1]),np.log(15),np.log(300)],method='L-BFGS-B',bounds=[(np.log(0.01),np.log(50))]*4+[(np.log(0.1),np.log(500)),(np.log(10),np.log(1000))])
    top=np.argsort(ll)[::-1]
    peaks=[]
    for j in top:
        if all(abs(np.log(Ps[j]/p))>0.05 for p,_ in peaks): peaks.append((Ps[j],ll[j]))
        if len(peaks)>=6: break
    key='joint' if joint else 'dLW_103'
    out[key]=dict(P_best=float(Ps[i]),lnL_best=float(ll[i]),w_best=float(ww[i]),lnL_SE=float(-se.fun),dlnL_vs_SE=float(ll[i]+se.fun),
                  peaks=[(round(float(p),2),round(float(l-ll[i]),2)) for p,l in peaks],grid=Ps.tolist(),lnL=ll.tolist())
    print(key,{k:v for k,v in out[key].items() if k not in ('grid','lnL')},flush=True)
json.dump(out,open('qp_profile.json','w'))
