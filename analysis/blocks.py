# Four equal-length blocks: amplitude and phase at P0 with per-label jitters fixed at global values.
import numpy as np
from scipy.stats import chi2
exec(open('split.py').read().split('subsets={')[0])
def blocks(df,mode):
    X=np.c_[np.cos(2*np.pi*df.t/P0),np.sin(2*np.pi*df.t/P0)]
    jn=jitter_fit(df,design=X)[0] if mode=='1p' else jitter_fit(df)[0]
    labs_all=[l for l in LABS if (df.lab==l).any()]
    def fit(b):
        labs=[l for l in LABS if (b.lab==l).any()]; idx=np.array([labs.index(l) for l in b.lab])
        jj=np.array([jn[labs_all.index(l)] for l in labs]); s2=b.e.values**2+jj[idx]**2; w=1/s2
        L=np.array([[1.0*(l==x) for x in labs] for l in b.lab]); XX=np.hstack([L,np.c_[np.cos(2*np.pi*b.t/P0),np.sin(2*np.pi*b.t/P0)]])
        A=XX.T@(XX*w[:,None]); p=np.linalg.solve(A,XX.T@(w*b.y.values)); C=np.linalg.inv(A)
        a,bb=p[-2:]; K=np.hypot(a,bb); ph=np.degrees(np.arctan2(bb,a)); J=np.array([a/K,bb/K]); sK=np.sqrt(J@C[-2:,-2:]@J)
        Jp=np.array([-bb,a])/K**2; sph=np.degrees(np.sqrt(Jp@C[-2:,-2:]@Jp)); return K,sK,ph,sph
    Kg,sKg,phg,sphg=fit(df)
    e=np.linspace(df.t.min(),df.t.max()+1e-6,5); rows=[]
    for k in range(4):
        b=df[(df.t>=e[k])&(df.t<e[k+1])]; K,sK,ph,sph=fit(b); dph=(ph-phg+180)%360-180
        rows.append([np.mean(b.t),len(b),K,sK,dph,sph])
    r=np.array(rows)
    rms=np.sqrt(np.mean(r[:,4]**2)); c2p=np.sum((r[:,4]/r[:,5])**2); c2K=np.sum(((r[:,2]-Kg)/r[:,3])**2)
    return r,Kg,sKg,rms,chi2.sf(c2p,4),chi2.sf(c2K,4)
for mode in ['1p','0p']:
    r,Kg,sKg,rms,pp,pK=blocks(S,mode)
    print(mode,'global K %.2f±%.2f  phase rms %.1f  p_phase %.2f  p_K %.2f'%(Kg,sKg,rms,pp,pK)); print(np.round(r,1))
    if mode=='1p': np.save('blocks_104.npy',r); np.save('blocks_104_global.npy',np.array([Kg,sKg]))
