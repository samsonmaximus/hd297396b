# v15: forward model of strictly coherent rotation with P_rot ~ U(30,50) d (the adopted range); makes fwd_3050.npz for Fig. 6.
exec(open('split.py').read().split('subsets={')[0])
rng=np.random.default_rng(3051)
df=S; t=df.t.values; lab=df.lab.values
X=np.c_[np.cos(2*np.pi*t/P0),np.sin(2*np.pi*t/P0)]
jj,_,idx,L,labs=jitter_fit(df,design=X)
s2=df.e.values**2+jj[idx]**2; w=1/s2; XX=np.hstack([L,X]); p=np.linalg.solve(XX.T@(XX*w[:,None]),XX.T@(w*df.y.values)); res=df.y.values-XX@p
jnull=jitter_fit(df)[0]
f0=np.array([1/P0]); N=20000
def noise(n):
    R=np.empty((len(t),n))
    for k in range(n):
        rr=res.copy()
        for l in np.unique(lab):
            m=np.where(lab==l)[0]; rr[m]=res[rng.permutation(m)]
        R[:,k]=rr
    return R
def dchi(Y): return periodogram(df,f0,jit=jnull,Y=Y)[0]
out={}
out['noise']=dchi(noise(N))
def act(A,n):
    Pr=rng.uniform(30,50,n); Y=np.zeros((len(t),n))
    for h,a in [(1,1),(2,0.5),(3,1/3)]:
        Y+=A*a*np.sin(2*np.pi*h*t[:,None]/Pr[None,:]+rng.uniform(0,2*np.pi,n)[None,:])
    return Y+noise(n)
out['act']=dchi(act(4.5,N))
for A2 in [8,10,12,15,25]:
    z=dchi(act(A2,4000)); print('A=%d: frac >= 32.26: %.4f'%(A2,(z>=32.26).mean()))
Kp=np.hypot(*p[-2:])
out['kep']=dchi(Kp*np.sin(2*np.pi*t[:,None]/P0+rng.uniform(0,2*np.pi,N)[None,:])+noise(N))
print('K used %.2f'%Kp)
for k,v in out.items(): print(k,'99pct %.1f max %.1f frac>=32.26 %.4f'%(np.percentile(v,99),v.max(),(v>=32.26).mean()))
np.savez('fwd_3050.npz',**out)
