# Fig. 4 (v13): signal growth with time and four-block coherence.
import json
exec(open('split.py').read().split('subsets={')[0])
exec(open('style.py').read())
G=json.load(open('growth.json')); r=np.load('blocks_104.npy'); Kg,sKg=np.load('blocks_104_global.npy')
Ss=S.sort_values('t').reset_index(drop=True); S3=Ss[np.abs(Ss.t-2454922.53)>0.2].reset_index(drop=True)
fig,ax=plt.subplots(3,1,figsize=(W1,4.3),sharex=True,gridspec_kw=dict(height_ratios=[1.25,1,1]))
for key,df,c,lab in [('104',Ss,COL['pre_072'],'104 epochs'),('103',S3,COL['pre_oth'],'103 epochs')]:
    g=G[key]; n=np.array(g['n']); tt=df.t.values[n-1]-2450000
    ax[0].fill_between(tt,g['lo'],g['hi'],color=c,alpha=0.13,lw=0,step='post')
    ax[0].step(tt,g['lo'],where='post',color=c,lw=0.4,alpha=0.6); ax[0].step(tt,g['hi'],where='post',color=c,lw=0.4,alpha=0.6)
    ax[0].step(tt,g['obs'],where='post',color=c,lw=0.9,label=lab)
ax[0].set_ylabel(r'$\Delta\chi^2$ at $P_0$'); ax[0].set_ylim(0,55)
ax[0].legend(frameon=False,loc='upper left',fontsize=6.3)
ax[0].text(0.98,0.06,'bands: 90% range of a\nstationary Keplerian',transform=ax[0].transAxes,ha='right',fontsize=6.0,color='#555')
x=r[:,0]-2450000
ax[1].axhspan(Kg-sKg,Kg+sKg,color='#e4e4e0',lw=0); ax[1].axhline(Kg,color=INK,lw=0.6)
ax[1].errorbar(x,r[:,2],r[:,3],fmt='o',ms=3.5,color=COL['pre_072'],elinewidth=0.8,capsize=0)
for xi,n_,Ki,s in zip(x,r[:,1],r[:,2],r[:,3]): ax[1].text(xi,Ki+s+0.4,f'n = {int(n_)}',ha='center',fontsize=6.3)
ax[1].set_ylabel(r'$K$ (m s$^{-1}$)'); ax[1].set_ylim(0,10.8)
ax[2].axhspan(-104,104,color='#f0f0ed',lw=0); ax[2].axhline(0,color=INK,lw=0.6)
ax[2].errorbar(x,r[:,4],r[:,5],fmt='o',ms=3.5,color=COL['pre_072'],elinewidth=0.8,capsize=0)
ax[2].set_ylabel(r'phase $-$ global (deg)'); ax[2].set_xlabel(r'BJD $-$ 2450000'); ax[2].set_ylim(-125,125)
ax[2].text(0.02,0.05,r'rms 15$^\circ$; random phases would give 104$^\circ$ (shaded)',transform=ax[2].transAxes,fontsize=6.3,color='#555')
ax[2].set_xlim(2900,9700)
for a,l in zip(ax,'abc'): a.text(0.985,0.95,f'({l})',transform=a.transAxes,va='top',ha='right',fontsize=7,bbox=dict(fc='white',ec='none',pad=0.8))
fig.subplots_adjust(hspace=0.07); fig.savefig('fig_coherence.pdf'); fig.savefig('fig_coherence.png'); plt.close()
print('ok')
