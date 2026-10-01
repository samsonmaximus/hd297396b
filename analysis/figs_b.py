exec(open('split.py').read().split('subsets={')[0])
exec(open('style.py').read())
# --- coherence figure from blocks_104
r=np.load('blocks_104.npy'); Kg=5.75; sKg=0.58
fig,ax=plt.subplots(2,1,figsize=(W1,2.9),sharex=True)
x=r[:,0]-2450000
ax[0].axhspan(Kg-sKg,Kg+sKg,color='#e4e4e0',lw=0); ax[0].axhline(Kg,color=INK,lw=0.6)
ax[0].errorbar(x,r[:,2],r[:,3],fmt='o',ms=3.5,color=COL['pre_072'],elinewidth=0.8,capsize=0)
for xi,n,Ki,s in zip(x,r[:,1],r[:,2],r[:,3]): ax[0].text(xi,Ki+s+0.35,f'n = {int(n)}',ha='center',fontsize=6.5)
ax[0].set_ylabel(r'$K$ (m s$^{-1}$)'); ax[0].set_ylim(0,10.5)
ax[1].axhspan(-104,104,color='#f0f0ed',lw=0); ax[1].axhline(0,color=INK,lw=0.6)
ax[1].errorbar(x,r[:,4],r[:,5],fmt='o',ms=3.5,color=COL['pre_072'],elinewidth=0.8,capsize=0)
ax[1].set_ylabel(r'phase $-$ global (deg)'); ax[1].set_xlim(3500,9100); ax[1].set_xlabel(r'BJD $-$ 2450000'); ax[1].set_ylim(-125,125)
ax[1].text(0.02,0.05,r'rms 15$^\circ$; random phases would give 104$^\circ$ (shaded)',transform=ax[1].transAxes,fontsize=6.5,color='#555')
fig.subplots_adjust(hspace=0.07); fig.savefig('fig_coherence.pdf'); fig.savefig('fig_coherence.png'); plt.close()
# --- forward model figure
F=np.load('fwd.npz')
fig,ax=plt.subplots(figsize=(W1,2.3))
bins=np.linspace(0,110,111)
for k,c,l in [('noise',MUTED,'noise only'),('act',ACC,r'coherent rotation, $A=4.5$ m s$^{-1}$'),('kep',COL['pre_072'],r'Keplerian, $K=5.7$ m s$^{-1}$')]:
    ax.hist(np.ravel(F[k]),bins=bins,histtype='step',density=True,color=c,lw=1.0,label=l)
ax.axvline(32.26,color=INK,lw=0.9); ax.text(33.5,2e-5,'observed',rotation=90,fontsize=6.5,va='bottom')
ax.set_yscale('log'); ax.set_ylim(1e-5,1); ax.set_xlim(0,110)
ax.set_xlabel(r'$\Delta\chi^2$ at $P_0$'); ax.set_ylabel('density'); ax.legend(frameon=False,loc='upper right')
fig.savefig('fig_forward.pdf'); fig.savefig('fig_forward.png'); plt.close()
# --- held-out figure
S103=S[np.abs(S.t-2454922.53)>0.2]
A=S103[S103.lab=='pre_072']; B=S[S.lab!='pre_072']
zA,_=periodogram(A,freqs); zB,_=periodogram(B,freqs)
PA=4.26920; sPA=0.00101
fig=plt.figure(figsize=(W2,2.5))
gs=fig.add_gridspec(1,3,width_ratios=[1.6,1,1],wspace=0.32)
a0=fig.add_subplot(gs[0])
a0.plot(1/freqs,zA,color=COL['pre_072'],lw=0.5,label='072.C-0488 only (46 epochs, 2004–09)')
a0.plot(1/freqs,-zB,color=COL['pre_oth'],lw=0.5,label='all other programmes (57 epochs, 2004–21)')
a0.axvline(P0,color=INK,lw=0.5,ls=':'); a0.axhline(0,color=INK,lw=0.4)
a0.set_xscale('log'); a0.set_xlim(1.05,1000); a0.set_xlabel('period (d)'); a0.set_ylabel(r'$\Delta\chi^2$ (lower set plotted negative)')
a0.set_ylim(-26,44); a0.legend(frameon=False,loc='upper right',fontsize=6.0)
for k,(df,c,title) in enumerate([(A,COL['pre_072'],'072.C-0488 only'),(B,COL['pre_oth'],'all other programmes')]):
    ax=fig.add_subplot(gs[k+1])
    X=np.c_[np.cos(2*np.pi*df.t/P0),np.sin(2*np.pi*df.t/P0)]
    jj,_,idx,L,labs=jitter_fit(df,design=X); s=np.sqrt(df.e.values**2+jj[idx]**2); w=1/s**2
    XX=np.hstack([L,X]); p=np.linalg.solve(XX.T@(XX*w[:,None]),XX.T@(w*df.y.values))
    y=df.y.values-L@p[:len(labs)]
    phs=((df.t.values-2455000)/P0)%1
    ax.errorbar(phs,y,s,fmt='o',ms=2.5,color=c,alpha=0.75,elinewidth=0.5,capsize=0)
    tt=np.linspace(0,1,200); m=p[-2]*np.cos(2*np.pi*(tt*P0+2455000)/P0)+p[-1]*np.sin(2*np.pi*(tt*P0+2455000)/P0)
    ax.plot(tt,m,color=INK,lw=1)
    K=np.hypot(*p[-2:]); ax.set_title(f'{title}: $K$ = {K:.1f} m s$^{{-1}}$',fontsize=7)
    ax.set_xlabel('phase at $P_0$'); ax.set_ylim(-22,22); ax.set_xlim(0,1)
    if k==0: ax.set_ylabel(r'RV $-$ offset (m s$^{-1}$)')
fig.savefig('fig_heldout.pdf'); fig.savefig('fig_heldout.png'); plt.close()
print('done')
