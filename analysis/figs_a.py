exec(open('split.py').read().split('subsets={')[0])
exec(open('style.py').read())
S103=S[np.abs(S.t-2454922.53)>0.2]
# ---------- Fig 2 periodogram stack
z,jn=periodogram(S,freqs)
X=np.c_[np.cos(2*np.pi*S.t/P0),np.sin(2*np.pi*S.t/P0)]
jj,_,idx,L,labs=jitter_fit(S,design=X)
s2=S.e.values**2+jj[idx]**2; w=1/s2; XX=np.hstack([L,X]); p=np.linalg.solve(XX.T@(XX*w[:,None]),XX.T@(w*S.y.values))
R=S.copy(); R['y']=S.y.values-X@p[-2:]
zr,_=periodogram(R,freqs)
t=S.t.values; ph=2*np.pi*np.outer(t,freqs); Wn=np.abs(np.exp(1j*ph).sum(0))**2/len(t)**2
fig,ax=plt.subplots(3,1,figsize=(W1,4.2),sharex=True)
per=1/freqs
def binmax(x,y,nb=1400):
    lb=np.log10(x); e=np.linspace(lb.min(),lb.max(),nb+1); i=np.clip(np.digitize(lb,e)-1,0,nb-1)
    m=np.full(nb,np.nan); xm=np.full(nb,np.nan)
    for k in range(nb):
        sel=i==k
        if sel.any(): j=np.argmax(y[sel]); m[k]=y[sel][j]; xm[k]=x[sel][j]
    o=np.argsort(xm); return xm[o],m[o]
for a,y,lab in [(ax[0],z,'velocities'),(ax[1],zr,r'residuals, $P_0$ removed')]:
    xb,yb=binmax(per,y); a.plot(xb,yb,color=INK,lw=0.5); a.set_ylabel(r'$\Delta\chi^2$'); a.text(0.25,0.86,lab,transform=a.transAxes)
    a.axhline(28.3,color=MUTED,ls='--',lw=0.6)
ax[0].text(1000,29.2,'1% global FAP',ha='right',color=MUTED,fontsize=6.3)
ax[2].plot(per,Wn,color=INK,lw=0.5); ax[2].set_ylabel('window power'); ax[2].set_xlabel('period (d)')
for a in ax[:2]:
    a.set_ylim(0,40)
    for P,c in [(P0,COL['pre_072']),(1.30131,ACC),(200.886,COL['pre_oth'])]: a.plot([P],[38],marker='v',ms=4,color=c,ls='none')
xb,yb=binmax(per,Wn); ax[2].cla(); ax[2].plot(xb,yb,color=INK,lw=0.5); ax[2].set_ylabel('window power'); ax[2].set_xlabel('period (d)')
for P,c in [(P0,COL['pre_072']),(1.30131,ACC),(200.886,COL['pre_oth'])]: ax[2].plot([P],[0.45],marker='v',ms=4,color=c,ls='none')
ax[2].set_ylim(0,0.5)
ax[0].set_xscale('log'); ax[0].set_xlim(1.05,1000)
ax[0].text(P0*1.08,33,r'$P_0$',fontsize=6.5,color=COL['pre_072']); ax[0].text(1.36,26,'1.301 d',fontsize=6.5,color=ACC)
ax[1].text(215,25,'200.9 d',fontsize=6.5,color=COL['pre_oth'])
ax[2].text(365.25*1.1,0.36,'1 yr',fontsize=6.5,color=MUTED)
fig.subplots_adjust(hspace=0.06); fig.savefig('fig_periodogram.pdf'); fig.savefig('fig_periodogram.png')
print('residual max',1/freqs[np.argmax(zr)],zr.max())
