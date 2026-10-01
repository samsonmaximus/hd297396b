exec(open('split.py').read().split('subsets={')[0])
rng=np.random.default_rng(1)
S103=S[np.abs(S.t-2454922.53)>0.2]
train=S103[S103.lab=='pre_072']; test=S[S.lab!='pre_072']
# period and phase from training only
ztr,_=periodogram(train,freqs); Ptr=1/freqs[np.argmax(ztr)]
# refine
fine=np.linspace(1/Ptr-0.2/T,1/Ptr+0.2/T,801); zf,_=periodogram(train,fine); Ptr=1/fine[np.argmax(zf)]
Ktr,sKtr,phtr,sphtr,_=fitK(train,Ptr)
print('train-only period %.5f K %.2f±%.2f phase %.1f±%.1f'%(Ptr,Ktr,sKtr,phtr,sphtr))
# Period uncertainty from training alone; propagate phase drift to test epochs
Kte,sKte,phte,sphte,jte=fitK(test,Ptr)
print('test at train period: K %.2f±%.2f phase %.1f±%.1f'%(Kte,sKte,phte,sphte))
zt,jj=periodogram(test,np.array([1/Ptr]))
print('test dchi2 at train period (2 dof): %.2f  -> analytic p=%.2e'%(zt[0],np.exp(-zt[0]/2)))
# Monte Carlo null for test set: Gaussian with fitted null jitter, and within-label permutation of residuals-from-offsets
jnull,_,idx,L,labs=jitter_fit(test)
s=np.sqrt(test.e.values**2+jnull[idx]**2)
Y=rng.normal(size=(len(test),20000))*s[:,None]
zg,_=periodogram(test,np.array([1/Ptr]),jit=jnull,Y=Y)
print('Gaussian MC p = %.2e (%d/20000)'%((zg>=zt[0]).mean(),(zg>=zt[0]).sum()))
# permutation within label (keeps actual value distribution incl. outliers)
y=test.y.values.copy(); lab=test.lab.values
Yp=np.empty((len(y),20000))
for k in range(20000):
    yy=y.copy()
    for l in np.unique(lab):
        m=np.where(lab==l)[0]; yy[m]=y[rng.permutation(m)]
    Yp[:,k]=yy
zp,_=periodogram(test,np.array([1/Ptr]),jit=jnull,Y=Yp)
print('permutation p = %.2e (%d/20000)'%((zp>=zt[0]).mean(),(zp>=zt[0]).sum()))
# phase-directed statistic: projection onto predicted phase
dphi=(phte-phtr+180)%360-180
print('phase difference test-train: %.1f deg (combined sigma %.1f)'%(dphi,np.hypot(sphte,sphtr)))
# Period uncertainty -> phase prediction uncertainty at test midpoint
# training period uncertainty from Delta chi2 = 1 drop
fine=np.linspace(1/Ptr-1.0/T,1/Ptr+1.0/T,4001); zf,_=periodogram(train,fine)
ok=fine[zf>=zf.max()-1]; sigf=(ok.max()-ok.min())/2; sigP=sigf*Ptr**2
print('train period %.5f ± %.5f d'%(Ptr,sigP))
win=np.linspace(1/Ptr-3*sigf,1/Ptr+3*sigf,301)
ztw,_=periodogram(test,win); i=np.argmax(ztw)
print('test max in train 3-sigma window: dchi2 %.2f at P=%.5f'%(ztw[i],1/win[i]))
zgw,_=periodogram(test,win,jit=jnull,Y=Y[:,:20000]); mg=zgw.max(0)
zpw,_=periodogram(test,win,jit=jnull,Y=Yp[:,:20000]); mp=zpw.max(0)
print('window-max null: Gaussian p=%.2e (%d), permutation p=%.2e (%d)'%((mg>=ztw[i]).mean(),(mg>=ztw[i]).sum(),(mp>=ztw[i]).mean(),(mp>=ztw[i]).sum()))
Kte2,sK2,ph2,sph2,_=fitK(test,1/win[i]); print('test K at its window peak %.2f±%.2f'%(Kte2,sK2))
# reverse direction: train on non-GTO, test on GTO(46)
tr2=test; te2=train
z2,_=periodogram(tr2,freqs); f2=freqs[np.argmax(np.where(abs(freqs-1/4.268)<0.01,z2,0))]
fine=np.linspace(f2-1/T,f2+1/T,4001); zf,_=periodogram(tr2,fine); f2=fine[np.argmax(zf)]
ok=fine[zf>=zf.max()-1]; s2=(ok.max()-ok.min())/2
win2=np.linspace(f2-3*s2,f2+3*s2,301); zz,_=periodogram(te2,win2)
jn2,_,idx2,_,_=jitter_fit(te2); ss=np.sqrt(te2.e.values**2+jn2[idx2]**2)
Y2=rng.normal(size=(len(te2),20000))*ss[:,None]; zn,_=periodogram(te2,win2,jit=jn2,Y=Y2)
print('reverse: nonGTO-derived P=%.5f±%.5f; GTO(46) window max %.2f; Gaussian p=%.2e'%(1/f2,s2/f2**2,zz.max(),(zn.max(0)>=zz.max()).mean()))
print('note: nonGTO global periodogram rank of P0 peak:', (z2>z2[np.argmin(abs(freqs-f2))]).sum(),'grid pts higher; global max P=%.4f'%(1/freqs[np.argmax(z2)]))
