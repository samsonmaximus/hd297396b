exec(open('split.py').read().split('subsets={')[0])
rng=np.random.default_rng(3)
S103=S[np.abs(S.t-2454922.53)>0.2]
df=S103
jn,_,idx,L,labs=jitter_fit(df, design=np.c_[np.cos(2*np.pi*df.t/P0),np.sin(2*np.pi*df.t/P0)])
s=np.sqrt(df.e.values**2+jn[idx]**2)
fr=np.array([1/4.26838,1/1.30131]); fk=[]
for P in fr: 
    k=np.argmin(abs(freqs-P)); fk.append(freqs[k-4:k+5])
grid=np.concatenate(fk)
zo,_=periodogram(df,grid,jit=jn); obs=zo[:9].max()-zo[9:].max()
print('observed dchi2(P0)-dchi2(1.301) on 103: %.2f'%obs)
t=df.t.values
for Ptrue,K in [(4.26838,5.3),(1.30131,5.3)]:
    ND=4000
    ph=rng.uniform(0,2*np.pi,ND)
    Y=K*np.sin(2*np.pi*t[:,None]/Ptrue+ph[None,:])+rng.normal(size=(len(t),ND))*s[:,None]
    zz,_=periodogram(df,grid,jit=jn,Y=Y); diff=zz[:9].max(0)-zz[9:].max(0)
    print('truth %.3f: median diff %.2f, 5-95%% [%.2f, %.2f], frac with diff>=obs %.3f, frac diff>0 %.3f'%(Ptrue,np.median(diff),*np.percentile(diff,[5,95]),(diff>=obs).mean(),(diff>0).mean()))
