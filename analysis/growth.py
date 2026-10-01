# Growth of the signal with the number of epochs, in time order (cf. Sreenivas et al. 2022, Figs. 3, 6;
# Mortier & Collier Cameron 2017). Statistic: Delta chi^2 at P0 with the per-label offsets refit and the
# per-label jitters held at their full-data planet-free values. Expectation band: a Keplerian of the
# fitted amplitude and phase added to the fit residuals permuted within each label (2000 realisations).
import numpy as np, json
exec(open('split.py').read().split('subsets={')[0])
rng=np.random.default_rng(77)
out={}
for name,df in [('104',S),('103',S[np.abs(S.t-2454922.53)>0.2].reset_index(drop=True))]:
    df=df.sort_values('t').reset_index(drop=True); t=df.t.values; lab=df.lab.values
    jn,_,idx,L,labs=jitter_fit(df); s=np.sqrt(df.e.values**2+jn[idx]**2)
    X=np.c_[np.cos(2*np.pi*t/P0),np.sin(2*np.pi*t/P0)]
    w=1/s**2; XX=np.hstack([L,X]); p=np.linalg.solve(XX.T@(XX*w[:,None]),XX.T@(w*df.y.values)); res=df.y.values-XX@p
    model=X@p[-2:]
    def dchi(Y,n):
        # Delta chi^2 at P0 on first n epochs, labels present so far, fixed jitters
        sub=slice(0,n); Ls=L[sub]; keep=Ls.sum(0)>0; Ls=Ls[:,keep]
        ww=1/s[sub]; A=Ls*ww[:,None]; Q,_=np.linalg.qr(A); Pm=np.eye(n)-Q@Q.T
        Xs=Pm@(X[sub]*ww[:,None]); yw=Pm@(Y[sub]*ww[:,None] if Y.ndim==2 else Y[sub]*ww)
        F=Xs.T@Xs; b=Xs.T@yw
        return np.einsum('i...,ij,j...->...',b,np.linalg.inv(F),b)
    ns=np.arange(15,len(df)+1)
    obs=np.array([dchi(df.y.values,n) for n in ns])
    NS=2000; Ysim=np.empty((len(df),NS))
    for k in range(NS):
        rr=res.copy()
        for l in np.unique(lab):
            m=np.where(lab==l)[0]; rr[m]=res[rng.permutation(m)]
        Ysim[:,k]=model+rr
    sim=np.array([dchi(Ysim,n) for n in ns])
    lo,med,hi=np.percentile(sim,[5,50,95],axis=1)
    # fraction of Ns at which obs lies inside the 90% band; and a global conformity statistic
    inside=np.mean((obs>=lo)&(obs<=hi))
    # rank of the observed curve's mean squared standardized deviation among simulations
    mu=sim.mean(1); sd=sim.std(1)
    dev=lambda c: np.mean(((c-mu)/sd)**2)
    dsim=np.array([dev(sim[:,k]) for k in range(NS)]); dobs=dev(obs)
    pconf=(dsim>=dobs).mean()
    # slope of obs vs n (linear fit) vs expectation
    out[name]=dict(n=ns.tolist(),obs=obs.tolist(),lo=lo.tolist(),med=med.tolist(),hi=hi.tolist(),frac_inside=float(inside),p_conform=float(pconf),final=float(obs[-1]))
    print(name,'final %.1f  frac of N inside 90%% band %.2f  conformity p %.2f  obs at N=30,50,70,90: %s  median sim: %s'%(obs[-1],inside,pconf,np.round(obs[[15,35,55,75]],1),np.round(med[[15,35,55,75]],1)))
json.dump(out,open('growth.json','w'))
