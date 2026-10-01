# Robustness of the rotation-band tests to the adopted band: repeat over 22-40 d (Noyes range 22.7-36.0 d).
# (1) harmonic-band scan of the residuals (bands 22-40, 11-20, 7.33-13.33 d) with band-restricted 1% levels;
# (2) 90% recovery amplitude for a sinusoid anywhere in 22-40 d;
# (3) forward model of strictly coherent rotation (fundamental + 2 harmonics) with P_rot ~ U(22,40).
import numpy as np, json
exec(open('split.py').read().split('subsets={')[0])
rng=np.random.default_rng(1745)
df=S; t=df.t.values; lab=df.lab.values
X=np.c_[np.cos(2*np.pi*t/P0),np.sin(2*np.pi*t/P0)]
jj,_,idx,L,labs=jitter_fit(df,design=X); w=1/(df.e.values**2+jj[idx]**2); XX=np.hstack([L,X]); p=np.linalg.solve(XX.T@(XX*w[:,None]),XX.T@(w*df.y.values))
R=df.copy(); R['y']=df.y.values-X@p[-2:]; res=df.y.values-XX@p
jn,_,idxR,_,_=jitter_fit(R); s=np.sqrt(R.e.values**2+jn[idxR]**2)
Y=rng.normal(size=(len(R),4000))*s[:,None]; out={}
for nm,(lo,hi) in {'17-45':(17,45),'8.5-22.5':(8.5,22.5),'5.67-15':(5.667,15)}.items():
    f=np.arange(1/hi,1/lo,1/(5*6536)); z,_=periodogram(R,f); zz,_=periodogram(R,f,jit=jn,Y=Y); thr=np.percentile(zz.max(0),99)
    out[nm]=dict(max=float(z.max()),at=float(1/f[np.argmax(z)]),thr1=float(thr)); print('band',nm,out[nm],flush=True)
    if nm=='17-45': f22,thr22=f,thr
for K in [3.5,4.0,4.5,5.0]:
    n=400; Pi=rng.uniform(17,45,n); ph=rng.uniform(0,2*np.pi,n)
    Yi=R.y.values[:,None]+K*np.sin(2*np.pi*t[:,None]/Pi[None,:]+ph[None,:])
    z,_=periodogram(R,f22,jit=jn,Y=Yi); out[f'rec_{K}']=float((z.max(0)>=thr22).mean()); print('K=%.1f recovered %.3f'%(K,out[f'rec_{K}']),flush=True)
jnull=jitter_fit(df)[0]; f0=np.array([1/P0])
def noise(n):
    Rm=np.empty((len(t),n))
    for k in range(n):
        rr=res.copy()
        for l in np.unique(lab):
            m=np.where(lab==l)[0]; rr[m]=res[rng.permutation(m)]
        Rm[:,k]=rr
    return Rm
def act(A,n):
    Pr=rng.uniform(17,45,n); Ya=np.zeros((len(t),n))
    for h,a in [(1,1),(2,0.5),(3,1/3)]:
        Ya+=A*a*np.sin(2*np.pi*h*t[:,None]/Pr[None,:]+rng.uniform(0,2*np.pi,n)[None,:])
    return Ya+noise(n)
z=periodogram(df,f0,jit=jnull,Y=act(4.5,20000))[0]
out['fwd_4.5']=dict(frac=float((z>=32.26).mean()),p99=float(np.percentile(z,99)),max=float(z.max())); print('fwd 4.5',out['fwd_4.5'],flush=True)
for A in [12,15]:
    z=periodogram(df,f0,jit=jnull,Y=act(A,4000))[0]; out[f'fwd_{A}']=float((z>=32.26).mean()); print('fwd',A,out[f'fwd_{A}'],flush=True)
json.dump(out,open('rot1745.json','w'),indent=1)
