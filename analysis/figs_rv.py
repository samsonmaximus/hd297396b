exec(open('split.py').read().split('subsets={')[0])
exec(open('style.py').read())
P=4.26837; K=5.52; Tc=2456298.337
df=S.copy(); t=df.t.values
mod=lambda tt: -K*np.sin(2*np.pi*(tt-Tc)/P)
labs=[l for l in LABS]; L=np.array([[1.0*(l==x) for x in labs] for l in df.lab])
X=np.c_[np.cos(2*np.pi*t/P),np.sin(2*np.pi*t/P)]
jj,_,idx,_,_=jitter_fit(df,design=X); s=np.sqrt(df.e.values**2+jj[idx]**2); w=1/s**2
off=np.linalg.solve(L.T@(L*w[:,None]),L.T@(w*(df.y.values-mod(t))))
y=df.y.values-L@off
fig=plt.figure(figsize=(W2,4.3)); gs=fig.add_gridspec(2,2,height_ratios=[1,1.15],hspace=0.35,wspace=0.28)
a=fig.add_subplot(gs[0,:])
for l in LABS:
    m=df.lab.values==l
    a.errorbar(t[m]-2450000,y[m],s[m],fmt=MRK[l],ms=3,color=COL[l],elinewidth=0.5,capsize=0,label=NAME[l],alpha=0.9)
i9=np.argmin(abs(t-2454922.53)); a.plot([t[i9]-2450000],[17.6],marker='^',ms=5,color=COL['pre_072']); a.text(t[i9]-2450000+80,15.6,r'$+52\,$m s$^{-1}$ (off scale)',fontsize=6.5)
a.set_ylim(-20,19); a.set_xlabel(r'BJD $-$ 2450000'); a.set_ylabel(r'RV $-$ offset (m s$^{-1}$)')
a.legend(ncol=4,frameon=False,loc='upper left',fontsize=6.3,bbox_to_anchor=(0,1.18))
ph=((t-Tc)/P)%1
b=fig.add_subplot(gs[1,0]); c=fig.add_subplot(gs[1,1],sharey=b)
for l in LABS:
    m=df.lab.values==l
    b.errorbar(ph[m],y[m],s[m],fmt=MRK[l],ms=2.5,color=COL[l],elinewidth=0.4,capsize=0,alpha=0.55)
    c.errorbar(ph[m],y[m]-mod(t[m]),s[m],fmt=MRK[l],ms=2.5,color=COL[l],elinewidth=0.4,capsize=0,alpha=0.55)
pp=np.linspace(0,1,300); b.plot(pp,-K*np.sin(2*np.pi*pp),color=INK,lw=1.1)
edges=np.linspace(0,1,11); ok=np.abs(t-2454922.53)>0.2
for lo,hi in zip(edges[:-1],edges[1:]):
    m=(ph>=lo)&(ph<hi)&ok; ww=w[m]; mu=np.sum(ww*y[m])/ww.sum(); se=1/np.sqrt(ww.sum())
    b.errorbar((lo+hi)/2,mu,se,fmt='s',ms=4,mfc='white',mec=INK,ecolor=INK,elinewidth=0.8,capsize=0,zorder=5)
c.axhline(0,color=INK,lw=0.6)
b.set_xlim(0,1); c.set_xlim(0,1); b.set_ylim(-20,19)
b.set_xlabel('orbital phase from conjunction'); c.set_xlabel('orbital phase from conjunction'); b.set_ylabel(r'RV (m s$^{-1}$)'); c.set_ylabel(r'residual (m s$^{-1}$)')
b.text(0.03,0.9,r'$P=4.26837$ d, $K=5.52$ m s$^{-1}$',transform=b.transAxes,fontsize=6.5)
fig.savefig('fig_rv.pdf'); fig.savefig('fig_rv.png')
