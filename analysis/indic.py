exec(open('split.py').read().split('subsets={')[0])
exec(open('style.py').read())
from scipy.stats import pearsonr
rng=np.random.default_rng(41)
dd=d[keep].copy(); dd['night']=np.floor(dd.BJD-0.196).astype(int)
inds=[('dLW','e_dLW',r'$\Delta$LW'),('Halpha','e_Halpha',r'H$\alpha$'),('NaD1','e_NaD1','Na I D1'),('NaD2','e_NaD2','Na I D2'),
      ('FWHMDRS',None,'CCF FWHM'),('Contrast',None,'CCF contrast'),('BIS',None,'BIS'),('CRX','e_CRX','CRX'),('RHKp','e_RHKp',r"$R'_{\rm HK}$")]
# RV residuals (white-noise fit) on S
X=np.c_[np.cos(2*np.pi*S.t/P0),np.sin(2*np.pi*S.t/P0)]
jj,_,idx,L,labs=jitter_fit(S,design=X); w=1/(S.e.values**2+jj[idx]**2); XX=np.hstack([L,X]); p=np.linalg.solve(XX.T@(XX*w[:,None]),XX.T@(w*S.y.values))
res=S.y.values-XX@p
rows=[]; curves=[]
for col,ecol,name in inds:
    out=[]
    for (n,l),g in dd.groupby(['night','lab']):
        v=g[col].values; ok=np.isfinite(v)
        if ecol: e=g[ecol].values; ok&=np.isfinite(e)&(e>0)
        if not ok.any(): out.append((np.sum(g.BJD)/len(g),np.nan,np.nan,l)); continue
        if ecol: ww=1/e[ok]**2; out.append((np.mean(g.BJD),np.sum(ww*v[ok])/ww.sum(),1/np.sqrt(ww.sum()),l))
        else: out.append((np.mean(g.BJD),np.mean(v[ok]),np.nan,l))
    I=pd.DataFrame(out,columns=['t','y','e','lab']).sort_values('t').reset_index(drop=True)
    if not ecol or col=='RHKp':
        sc=I.groupby('lab').y.transform(lambda v:1.4826*np.median(np.abs(v-np.median(v))))
        I['e']=np.where(np.isfinite(I.e),I.e,sc*0.5)
    mask=np.isfinite(I.y).values&np.isfinite(I.e).values
    J=I[mask].reset_index(drop=True)
    z,jn=periodogram(J,freqs)
    # null: within-label shuffles of (value,error) pairs
    Ys=[];Es=[]
    nd=100
    zmax=[]
    for k in range(nd):
        Jk=J.copy()
        for l in Jk.lab.unique():
            m=np.where(Jk.lab==l)[0]; pm=rng.permutation(m); Jk.loc[m,'y']=J.y.values[pm]; Jk.loc[m,'e']=J.e.values[pm]
        zk,_=periodogram(Jk,freqs[::2],jit=jn); zmax.append(zk.max())
    thr=np.percentile(zmax,99)
    kk=np.argmin(abs(freqs-1/P0)); zP=z[kk-3:kk+4].max()
    # correlation with RV residuals, indicator per-label means removed
    tt=S.t.values; ind_at=np.interp(tt,J.t.values,J.y.values) # align
    # exact match by time
    mt={round(a,4):b for a,b in zip(J.t.values,J.y.values)}
    Iv=np.array([mt.get(round(a,4),np.nan) for a in I.t.values])
    Iv_full=I.y.values; lab_full=I.lab.values
    ok=np.isfinite(Iv_full)
    Ic=Iv_full-pd.Series(Iv_full).groupby(lab_full).transform('mean').values
    r,pv=pearsonr(res[ok],Ic[ok])
    r0,_=pearsonr((S.y.values-np.mean(S.y.values))[ok],Iv_full[ok])
    rows.append((name,ok.sum(),zP/thr,r,pv,r0)); curves.append((name,z,thr))
    print(f'{name:12s} n={ok.sum():3d} power/1%={zP/thr:.2f} r={r:+.2f} p={pv:.3f}  r_no_offsets={r0:+.2f}',flush=True)
np.save('indic_rows.npy',np.array(rows,dtype=object),allow_pickle=True)
fig,axs=plt.subplots(9,1,figsize=(W1,7.2),sharex=True)
for a,(name,z,thr),row in zip(axs,curves,rows):
    a.plot(1/freqs,z,color=INK,lw=0.4); a.axhline(thr,color=MUTED,ls='--',lw=0.6); a.axvline(P0,color=COL['pre_072'],lw=0.6,alpha=0.7)
    a.text(0.01,0.72,name,transform=a.transAxes,fontsize=6.5); a.text(0.99,0.72,'%.2f'%row[2],transform=a.transAxes,ha='right',fontsize=6.5)
    a.set_yticks([]); a.set_ylim(0,max(z.max(),thr)*1.25)
axs[0].set_xscale('log'); axs[0].set_xlim(1.05,1000); axs[-1].set_xlabel('period (d)')
fig.subplots_adjust(hspace=0.08); fig.savefig('fig_indicators.pdf'); fig.savefig('fig_indicators.png')
