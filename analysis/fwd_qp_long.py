# Forward model with evolving, harmonic-rich rotational signals: quasi-periodic Gaussian-process
# realisations (P_rot ~ U(17,45) d, evolution time 1-5 rotations, harmonic complexity w ~ U(0.25,0.8),
# which puts power into high harmonics of P_rot), added to the fit residuals permuted within each label,
# sampled at the 104 epochs and passed through the same periodogram statistic at P0.
import numpy as np, json
exec(open('split.py').read().split('subsets={')[0])
rng=np.random.default_rng(4546)
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
def qp(n,A):
    Y=np.empty((len(t),n)); meta=[]
    for k in range(n):
        P=rng.uniform(17,45); lam=P*rng.uniform(5,10); w=rng.uniform(0.8,1.5)
        K=A*A*np.exp(-r*r/(2*lam*lam)-np.sin(np.pi*r/P)**2/(2*w*w))+1e-8*np.eye(len(t))
        Y[:,k]=np.linalg.cholesky(K)@rng.normal(size=len(t)); meta.append((P,lam,w))
    return Y,np.array(meta)
out={}
for A,N in [(3.2,10000),(6.0,10000)]:
    Ya,meta=qp(N,A); z=periodogram(df,f0,jit=jnull,Y=Ya+noise(N))[0]
    out[str(A)]=dict(N=N,frac=float((z>=32.26).mean()),n=int((z>=32.26).sum()),p99=float(np.percentile(z,99)),max=float(z.max()),
                     P_at_max=float(meta[np.argmax(z),0]))
    print('GP sd %.1f m/s:'%A,out[str(A)],flush=True)
json.dump(out,open('fwd_qp_long.json','w'),indent=1)
