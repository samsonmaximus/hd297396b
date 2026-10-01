# v15: evolving-spot forward model for the Mount-Wilson-scale rotation estimate (P_rot 30-50 d),
# plus the most adversarial case: P_rot within 1% of 9*P0 = 38.415 d with sharp harmonics
# (small w puts power into high harmonics), so that the ninth harmonic can fall on P0.
import numpy as np, json
exec(open('split.py').read().split('subsets={')[0])
rng=np.random.default_rng(3050)
df=S; t=df.t.values; lab=df.lab.values
X=np.c_[np.cos(2*np.pi*t/P0),np.sin(2*np.pi*t/P0)]
jj,_,idx,L,labs=jitter_fit(df,design=X); w_=1/(df.e.values**2+jj[idx]**2); XX=np.hstack([L,X]); p=np.linalg.solve(XX.T@(XX*w_[:,None]),XX.T@(w_*df.y.values)); res=df.y.values-XX@p
jnull=jitter_fit(df)[0]; f0=np.array([1/P0]); r=t[:,None]-t[None,:]
def noise(n):
    R=np.empty((len(t),n))
    for k in range(n):
        rr=res.copy()
        for l in np.unique(lab):
            m=np.where(lab==l)[0]; rr[m]=res[rng.permutation(m)]
        R[:,k]=rr
    return R
def qp(n,A,Plo,Phi,nlo,nhi,wlo,whi):
    Y=np.empty((len(t),n)); meta=[]
    for k in range(n):
        P=rng.uniform(Plo,Phi); lam=P*rng.uniform(nlo,nhi); w=rng.uniform(wlo,whi)
        K=A*A*np.exp(-r*r/(2*lam*lam)-np.sin(np.pi*r/P)**2/(2*w*w))+1e-8*np.eye(len(t))
        Y[:,k]=np.linalg.cholesky(K)@rng.normal(size=len(t)); meta.append((P,lam,w))
    return Y,np.array(meta)
out={}
cases={'band30-50_short':(30,50,1,5,0.25,0.8),'band30-50_long':(30,50,5,10,0.8,1.5),
       'ninth_harmonic':(38.415*0.99,38.415*1.01,1,10,0.15,0.4)}
for nm,(Plo,Phi,nlo,nhi,wlo,whi) in cases.items():
    for A,N in [(3.2,10000),(6.0,10000),(10.0,5000)]:
        Ya,meta=qp(N,A,Plo,Phi,nlo,nhi,wlo,whi); z=periodogram(df,f0,jit=jnull,Y=Ya+noise(N))[0]
        key=f'{nm}_{A}'
        out[key]=dict(N=N,frac=float((z>=32.26).mean()),n=int((z>=32.26).sum()),p99=float(np.percentile(z,99)),max=float(z.max()))
        print(key,out[key],flush=True)
json.dump(out,open('fwd_qp_3050.json','w'),indent=1)
