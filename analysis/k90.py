exec(open('split.py').read().split('subsets={')[0])
rng=np.random.default_rng(22)
df=S; X=np.c_[np.cos(2*np.pi*df.t/P0),np.sin(2*np.pi*df.t/P0)]
jj,_,idx,L,labs=jitter_fit(df,design=X); w=1/(df.e.values**2+jj[idx]**2); XX=np.hstack([L,X]); p=np.linalg.solve(XX.T@(XX*w[:,None]),XX.T@(w*df.y.values))
R=df.copy(); R['y']=df.y.values-X@p[-2:]
jn,_,idxR,_,_=jitter_fit(R); s=np.sqrt(R.e.values**2+jn[idxR]**2)
f=np.arange(1/40,1/25,1/(5*6536))
zz,_=periodogram(R,f,jit=jn,Y=rng.normal(size=(len(R),4000))*s[:,None]); thr=np.percentile(zz.max(0),99)
for Kinj in [3.5,4.0,4.5,5.0,5.5,6.0]:
    n=400; Pi=rng.uniform(25,40,n); ph=rng.uniform(0,2*np.pi,n)
    Yi=R.y.values[:,None]+Kinj*np.sin(2*np.pi*R.t.values[:,None]/Pi[None,:]+ph[None,:])
    z,_=periodogram(R,f,jit=jn,Y=Yi); print('K=%.1f recovered %.3f (thr %.2f)'%(Kinj,(z.max(0)>=thr).mean(),thr),flush=True)
