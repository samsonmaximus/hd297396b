exec(open('split.py').read().split('subsets={')[0])
from scipy.stats import shapiro
rng=np.random.default_rng(21)
df=S; t=df.t.values
def wls(df,cols):
    jj,_,idx,L,labs=jitter_fit(df,design=cols)
    w=1/(df.e.values**2+jj[idx]**2); XX=np.hstack([L,cols]); A=XX.T@(XX*w[:,None]); p=np.linalg.solve(A,XX.T@(w*df.y.values)); C=np.linalg.inv(A)
    return p,C,jj,w,XX
X=lambda d,P: np.c_[np.cos(2*np.pi*d.t.values/P),np.sin(2*np.pi*d.t.values/P)]
# 1. jackknife K (white-noise, jitter refit)
K0=fitK(df)[0]; Ks=[]
for i in range(len(df)):
    Ks.append(fitK(df.drop(df.index[i]))[0])
Ks=np.array(Ks); dK=Ks-K0
i9=np.argmin(abs(df.t.values-2454922.53))
print('jackknife: K0 %.2f rms dK %.3f max |dK| %.3f at BJD %.2f; deleting discrepant: %.3f'%(K0,np.sqrt(np.mean(dK**2)),np.abs(dK).max(),df.t.values[np.argmax(abs(dK))],dK[i9]))
# 2. alias vs P0 at each single deletion
fa=np.array([1/P0,1/1.30131]); worst=1e9; meds=[]
fr0=np.linspace(1/P0-0.3/6536,1/P0+0.3/6536,31); fr1=np.linspace(1/1.30131-0.3/6536,1/1.30131+0.3/6536,31)
for i in range(len(df)):
    d2=df.drop(df.index[i]); z0,_=periodogram(d2,fr0); z1,_=periodogram(d2,fr1); dd=z0.max()-z1.max(); meds.append(dd); worst=min(worst,dd)
print('alias deletions: min %.2f median %.2f'%(worst,np.median(meds)))
# 3. residual 200.9 d
p,C,jj,w,XX=wls(df,X(df,P0)); R=df.copy(); R['y']=df.y.values-X(df,P0)@p[-2:]
fr=np.linspace(1/220,1/185,600); zr,_=periodogram(R,fr); i=np.argmax(zr); P2=1/fr[i]
K2,sK2,_,_,_=fitK(R,P2)
zrs=[]
for k in range(len(R)):
    z,_=periodogram(R.drop(R.index[k]),np.array([1/P2])); zrs.append(z[0])
print('200.9: P2 %.3f dchi2 %.2f K2 %.2f±%.2f min-after-deletion %.2f'%(P2,zr[i],K2,sK2,min(zrs)))
p2,C2,_,_,_=wls(df,np.hstack([X(df,P0),X(df,P2)]))
Kb2=np.hypot(*p2[-4:-2]); print('K_b with 200.9 fitted %.2f vs %.2f (sigma %.2f)'%(Kb2,K0,fitK(df)[1]))
# global FAP of residual peak
mx=np.load('mx_104_gauss.npy'); print('residual peak global FAP (104 null):',(mx>=zr[i]).mean())
# 4. TOI-6263.01 K limit, fixed ephemeris, white noise, planet b included
Pt=3.0178746
p3,C3,_,_,_=wls(df,np.hstack([X(df,P0),X(df,Pt)]))
a,b=p3[-2:]; Ct=C3[-2:,-2:]
draws=rng.multivariate_normal([a,b],Ct,200000); Kt=np.hypot(draws[:,0],draws[:,1])
print('TOI K %.2f; 95%% upper (posterior of amplitude, flat in a,b) %.2f'%(np.hypot(a,b),np.percentile(Kt,95)))
# 5. harmonic bands in residuals: band max, band-restricted 1% via Gaussian bootstrap, injection 90% recovery K in 25-40
bands={'25-40':(25,40),'12.5-20':(12.5,20),'8.3-13.3':(8.33,13.33)}
jn=jitter_fit(R)[0]; idxR=jitter_fit(R)[2]; s=np.sqrt(R.e.values**2+jn[idxR]**2)
Y=rng.normal(size=(len(R),4000))*s[:,None]
for nm,(lo,hi) in bands.items():
    f=np.arange(1/hi,1/lo,1/(5*6536)); z,_=periodogram(R,f); zz,_=periodogram(R,f,jit=jn,Y=Y); thr=np.percentile(zz.max(0),99)
    print('band %s: max %.2f at %.2f d; 1%% level %.2f'%(nm,z.max(),1/f[np.argmax(z)],thr))
    if nm=='25-40': f25,thr25=f,thr
for Kinj in [1.5,2.0,2.5,3.0]:
    n=300; Pi=rng.uniform(25,40,n); ph=rng.uniform(0,2*np.pi,n)
    Yi=R.y.values[:,None]+Kinj*np.sin(2*np.pi*R.t.values[:,None]/Pi[None,:]+ph[None,:])
    z,_=periodogram(R,f25,jit=jn,Y=Yi); print('  inject K=%.1f: recovered %.2f'%(Kinj,(z.max(0)>=thr25).mean()))
# 6. DRS jitters with planet
Kd=fitK(D); print('DRS K %.2f±%.2f jit %s ; SERVAL jit %s'%(Kd[0],Kd[1],np.round(Kd[4],2),np.round(fitK(S)[4],2)))
