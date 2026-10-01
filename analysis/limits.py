exec(open('split.py').read().split('subsets={')[0])
rng=np.random.default_rng(31)
df=S; X=np.c_[np.cos(2*np.pi*df.t/P0),np.sin(2*np.pi*df.t/P0)]
jj,_,idx,L,labs=jitter_fit(df,design=X); w=1/(df.e.values**2+jj[idx]**2); XX=np.hstack([L,X]); p=np.linalg.solve(XX.T@(XX*w[:,None]),XX.T@(w*df.y.values))
R=df.copy(); R['y']=df.y.values-X@p[-2:]
jn=jitter_fit(R)[0]
thr=np.percentile(np.load('mx_104_gauss.npy'),99)
Ps=np.geomspace(2,1000,36); Kg=np.array([1,1.5,2,2.5,3,3.5,4,5,6,7,8,10,12]); nph=60
out=np.zeros((len(Ps),len(Kg)))
t=R.t.values
for i,Pi in enumerate(Ps):
    f=np.linspace(1/Pi*0.97,1/Pi*1.03,61)
    for j,K in enumerate(Kg):
        ph=rng.uniform(0,2*np.pi,nph)
        Y=R.y.values[:,None]+K*np.sin(2*np.pi*t[:,None]/Pi+ph[None,:])
        z,_=periodogram(R,f,jit=jn,Y=Y); out[i,j]=(z.max(0)>=thr).mean()
K95=[np.interp(0.95,np.maximum.accumulate(out[i]),Kg) if out[i].max()>=0.95 else np.nan for i in range(len(Ps))]
K50=[np.interp(0.50,np.maximum.accumulate(out[i]),Kg) if out[i].max()>=0.5 else np.nan for i in range(len(Ps))]
np.savez('limits.npz',Ps=Ps,Kg=Kg,out=out,K95=K95,K50=K50,thr=thr)
for Pi,a,b in zip(Ps,K50,K95): print('%.1f  K50 %.2f K95 %.2f'%(Pi,a,b))
# at P0
f=np.linspace(1/P0*0.97,1/P0*1.03,61)
for K in [3,4,5,6,7]:
    ph=rng.uniform(0,2*np.pi,200); Y=R.y.values[:,None]+K*np.sin(2*np.pi*t[:,None]/P0+ph[None,:])
    z,_=periodogram(R,f,jit=jn,Y=Y); print('at P0 K=%d rec %.2f'%(K,(z.max(0)>=thr).mean()))
