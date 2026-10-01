# Is the 41.4-d quasi-periodicity in dLW stable? Profile likelihood over P in 20-100 d for: dLW first and
# second halves of the baseline, dLW with 072.C-0488 alone and without it, and H-alpha (full). 103 epochs.
import numpy as np, pandas as pd, json
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
exec(open('core.py').read().split("S=nightbin")[0])
dd=d[keep].copy(); dd['night']=np.floor(dd.BJD-0.196).astype(int)
rows=[]
for (n,l),gq in dd.groupby(['night','lab']):
    w=1/gq.e_dLW**2; wh=1/gq.e_Halpha**2
    rows.append(dict(t=np.mean(gq.BJD),lw=np.sum(w*gq.dLW)/w.sum(),elw=1/np.sqrt(w.sum()),ha=np.sum(wh*gq.Halpha)/wh.sum(),eha=1/np.sqrt(wh.sum()),lab=l))
A=pd.DataFrame(rows).sort_values('t').reset_index(drop=True); A=A[np.abs(A.t-2454922.53)>0.2].reset_index(drop=True)
LABS=['pre_072','pre_183','pre_oth','post']
def profile(sub,col,ecol,Ps):
    t=sub.t.values-2455000; r=t[:,None]-t[None,:]; labs=[l for l in LABS if (sub.lab==l).any()]; li=np.array([labs.index(l) for l in sub.lab])
    Lm=(li[:,None]==np.arange(len(labs))[None,:]).astype(float); y=sub[col].values*(1e3 if col=='ha' else 1); e=sub[ecol].values*(1e3 if col=='ha' else 1)
    def gll(jit,K0):
        K=K0.copy(); K[np.diag_indices_from(K)]+=e**2+jit[li]**2
        c=cho_factor(K,lower=True,check_finite=False); CiD=cho_solve(c,Lm,check_finite=False); Ciy=cho_solve(c,y,check_finite=False)
        p=np.linalg.solve(Lm.T@CiD,Lm.T@Ciy); rr=y-Lm@p
        return -0.5*rr@cho_solve(c,rr,check_finite=False)-np.sum(np.log(np.diag(c[0])))
    nl=len(labs); bnd=[(np.log(1e-3),np.log(100))]*nl+[(np.log(1e-3),np.log(500)),(np.log(10),np.log(1000)),(np.log(0.1),np.log(2))]
    sd=np.std(y)
    def fit(P):
        f=lambda x:-gll(np.exp(x[:nl]),np.exp(2*x[nl])*np.exp(-r*r/(2*np.exp(2*x[nl+1]))-np.sin(np.pi*r/P)**2/(2*np.exp(2*x[nl+2]))))
        best=None
        for w0 in [0.4,1.2]:
            x0=np.r_[np.log(np.full(nl,sd/3)),np.log(sd),np.log(300),np.log(w0)]
            rr=minimize(f,x0,method='L-BFGS-B',bounds=bnd)
            if best is None or rr.fun<best.fun: best=rr
        return -best.fun
    fse=lambda x:-gll(np.exp(x[:nl]),np.exp(2*x[nl])*np.exp(-r*r/(2*np.exp(2*x[nl+1]))))
    se=-minimize(fse,np.r_[np.log(np.full(nl,sd/3)),np.log(sd),np.log(300)],method='L-BFGS-B',bounds=bnd[:-1]).fun
    ll=np.array([fit(P) for P in Ps]); i=np.argmax(ll)
    near=ll[np.abs(np.log(Ps/41.4))<0.03].max()
    return dict(n=len(sub),P_best=float(Ps[i]),dlnL_best_vs_SE=float(ll[i]-se),dlnL_41_vs_SE=float(near-se),dlnL_41_vs_best=float(near-ll[i]))
Ps=np.exp(np.linspace(np.log(20),np.log(100),110)); out={}
tm=np.median(A.t)
for key,sub,col,ecol in [('dLW_first_half',A[A.t<tm],'lw','elw'),('dLW_second_half',A[A.t>=tm],'lw','elw'),
                         ('dLW_072_only',A[A.lab=='pre_072'],'lw','elw'),('dLW_not_072',A[A.lab!='pre_072'],'lw','elw'),
                         ('Halpha_all',A,'ha','eha')]:
    out[key]=profile(sub.reset_index(drop=True),col,ecol,Ps); print(key,out[key],flush=True)
# spectral window of the 103 epochs at 1/41.4 d
t=A.t.values; f=np.linspace(1/100,1/20,4000); W=np.abs(np.exp(2j*np.pi*np.outer(t,f)).sum(0))/len(t)
out['window_at_41.4']=float(np.interp(1/41.4,f,W)); out['window_pct']=float((W<np.interp(1/41.4,f,W)).mean()*100)
print('window at 41.4 d: %.3f (percentile %.0f within 20-100 d)'%(out['window_at_41.4'],out['window_pct']))
json.dump(out,open('qp_split.json','w'),indent=1)
