exec(open('pg.py').read())
import warnings,time; warnings.filterwarnings('ignore')
rng=np.random.default_rng(7)
def run(df,ND=10000,mode='gauss'):
    z,_=periodogram(df,freqs); obs=z.max()
    jn,_,idx,L,labs=jitter_fit(df); s=np.sqrt(df.e.values**2+jn[idx]**2)
    y=df.y.values; lab=df.lab.values
    mx=[]
    for b in range(ND//1000):
        if mode=='gauss': Y=rng.normal(size=(len(df),1000))*s[:,None]
        else:
            Y=np.empty((len(df),1000))
            for k in range(1000):
                yy=y.copy()
                for l in np.unique(lab):
                    m=np.where(lab==l)[0]; yy[m]=y[rng.permutation(m)]
                Y[:,k]=yy
        zz,_=periodogram(df,freqs,jit=jn,Y=Y); mx.append(zz.max(0))
    mx=np.concatenate(mx)
    return obs,mx
t0=time.time()
for name,df in [('104',S),('103',S[np.abs(S.t-2454922.53)>0.2])]:
    for mode in ['gauss','perm']:
        obs,mx=run(df,10000,mode)
        n=(mx>=obs).sum()
        print(name,mode,'obs %.2f  exceed %d/10000  p=%.1e  99.9pct=%.2f  max=%.2f  (%.0fs)'%(obs,n,max(n,0)/1e4,np.percentile(mx,99.9),mx.max(),time.time()-t0),flush=True)
        np.save(f'mx_{name}_{mode}.npy',mx)
