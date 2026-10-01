# v15: Fig. 6 from the 30-50 d forward model (fwd2_3050.py -> fwd_3050.npz); same layout as figs_b.py.
import numpy as np
exec(open('style.py').read())
F=np.load('fwd_3050.npz')
fig,ax=plt.subplots(figsize=(W1,2.3))
bins=np.linspace(0,110,111)
for k,c,l in [('noise',MUTED,'noise only'),('act',ACC,r'coherent rotation, $A=4.5$ m s$^{-1}$'),('kep',COL['pre_072'],r'Keplerian, $K=5.7$ m s$^{-1}$')]:
    ax.hist(np.ravel(F[k]),bins=bins,histtype='step',density=True,color=c,lw=1.0,label=l)
ax.axvline(32.26,color=INK,lw=0.9); ax.text(33.5,2e-5,'observed',rotation=90,fontsize=6.5,va='bottom')
ax.set_yscale('log'); ax.set_ylim(1e-5,1); ax.set_xlim(0,110)
ax.set_xlabel(r'$\Delta\chi^2$ at $P_0$'); ax.set_ylabel('density'); ax.legend(frameon=False,loc='upper right')
fig.savefig('../fig_forward.pdf'); fig.savefig('../fig_forward.png'); plt.close()
print('ok')
