import gzip, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.optimize import minimize
rows=[]
with gzip.open('data/table4.dat.gz','rt') as f:
    for l in f:
        try: rows.append((l[0:14].strip(), float(l[58:73]), float(l[74:87]), float(l[88:98])))
        except: pass
T4=pd.DataFrame(rows,columns=['star','bjd','rv','e'])
T4=T4[np.isfinite(T4.rv)&np.isfinite(T4.e)&(T4.e>0)&(T4.e<10)]
T4['night']=np.floor(T4.bjd-0.196).astype(int); T4['post']=T4.bjd>2457170
def nb(g):
    w=1/g.e**2
    return pd.Series(dict(t=(w*g.bjd).sum()/w.sum(), y=(w*g.rv).sum()/w.sum(), e=1/np.sqrt(w.sum())))
P0=4.26838; f0=1/P0
hd=set(T4[T4.star=='HD297396'].night)
def stat(t,y,e,lab,freqs,phi_ref=None):
    labs=np.unique(lab); L=(lab[:,None]==labs[None,:]).astype(float); idx=np.searchsorted(labs,lab)
    def nll(lj):
        s2=e**2+np.exp(lj)[idx]**2; w=1/s2; A=L.T@(L*w[:,None]); p=np.linalg.solve(A,L.T@(w*y)); r=y-L@p
        return 0.5*np.sum(r*r*w+np.log(s2))
    r=minimize(nll,np.ones(len(labs)),method='L-BFGS-B',bounds=[(-5,5)]*len(labs))
    s=np.sqrt(e**2+np.exp(r.x)[idx]**2); w=1/s
    Q,_=np.linalg.qr(L*w[:,None]); P=np.eye(len(t))-Q@Q.T; yp=P@(y*w)
    ph=2*np.pi*np.outer(t,freqs); C=P@(np.cos(ph)*w[:,None]); Sn=P@(np.sin(ph)*w[:,None])
    cc=(C*C).sum(0); ss=(Sn*Sn).sum(0); cs=(C*Sn).sum(0); a=C.T@yp; b=Sn.T@yp; det=cc*ss-cs*cs
    z=(ss*a*a-2*cs*a*b+cc*b*b)/det
    return z
res=[]
ctrl_f=f0+np.linspace(-0.02,0.02,41); ctrl_f=ctrl_f[np.abs(ctrl_f-f0)>0.002]
for star,g in T4.groupby('star'):
    if star=='HD297396': continue
    B=g.groupby(['night','post']).apply(nb).reset_index()
    if len(B)<30 or B.t.max()-B.t.min()<1000: continue
    shared=len(set(B.night)&hd)
    z=stat(B.t.values,B.y.values,B.e.values,B.post.values.astype(int),np.r_[f0,ctrl_f])
    res.append(dict(star=star,n=len(B),shared=shared,z0=z[0],zc_med=np.median(z[1:]),zc_max=z[1:].max()))
R=pd.DataFrame(res); R.to_csv('xstar.csv',index=False)
print('stars tested',len(R))
obs_hd=32.26  # our 104-epoch value (white noise, 4 labels)
print('fraction of stars with dchi2(P0) >= 32.26: %.4f (%d)'%((R.z0>=obs_hd).mean(),(R.z0>=obs_hd).sum()))
print('median dchi2 at P0 %.2f vs control freqs median %.2f (chi2_2 median = 1.39)'%(R.z0.median(),R.zc_med.median()))
print('mean dchi2 at P0 %.2f vs control %.2f'%(R.z0.mean(),R.zc_med.mean()))
for thr in [9.21,13.8]:
    print(' frac z0>%.2f: %.4f   (chi2_2 expects %.4f)'%(thr,(R.z0>thr).mean(),np.exp(-thr/2)))
Sh=R[R.shared>=10]
print('stars sharing >=10 nights with HD297396:',len(Sh),' median z0 %.2f, frac>9.21: %.3f'%(Sh.z0.median(),(Sh.z0>9.21).mean()))
print(R.sort_values('z0',ascending=False).head(10).to_string())
