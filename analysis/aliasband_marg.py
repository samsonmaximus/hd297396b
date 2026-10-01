# Alias comparison with all noise parameters marginalised (same machinery and proposal as fip.py):
# one-planet evidences in Jeffreys bands of equal log width centred on P0 and on the 1.30131-d alias,
# and with priors uniform in frequency over the same bands.
import numpy as np, json
src=open('fip.py').read()
exec(src.split("P0=4.26838\nres={}")[0].replace("NJ=int(sys.argv[1]) if len(sys.argv)>1 else 240","NJ=240"))
P0=4.26838; hw=0.5*np.log(4.34/4.20)
def band_lnZ(u,D,Pc,unif=False):
    labs,idx,L,t,y,e=D; w=1/np.sqrt(e**2+np.exp(u)[idx]**2)
    Q,_=np.linalg.qr(L*w[:,None]); P=np.eye(len(t))-Q@Q.T; yp=P@(y*w)
    f=np.linspace(1/(Pc*np.exp(hw)),1/(Pc*np.exp(-hw)),8001); d=f[1]-f[0]
    ph=2*np.pi*np.outer(t-t.mean(),f); C=P@(np.cos(ph)*w[:,None]); S_=P@(np.sin(ph)*w[:,None])
    lz,_=lnZratio_fast(np.c_[(C*C).sum(0),(S_*S_).sum(0),(C*S_).sum(0),C.T@yp,S_.T@yp])
    lp=np.log(d/(f[-1]-f[0])) if unif else np.log(d/(f*2*hw))
    return lse(lz+lp)
rng=np.random.default_rng(11); out={}
for name,df in [('103',S[np.abs(S.t-2454922.53)>0.2].reset_index(drop=True)),('104',S)]:
    D=setup(df)
    comps=[map_u(D),map_u(D,P0),map_u(D,P0,200.886)]
    qs=[multivariate_t(loc=m,shape=3.0*C,df=4) for m,C in comps]
    U=np.vstack([q.rvs(size=NJ//3,random_state=rng) for q in qs])
    lq=np.log(np.mean([np.exp(q.logpdf(U)) for q in qs],axis=0))
    inside=np.all((U>LNJ[0])&(U<LNJ[1]),axis=1); lpu=np.where(inside,-4*np.log(LNJ[1]-LNJ[0]),-np.inf)
    acc={k:[] for k in ['l0','P0','al','P0u','alu']}
    for n,u in enumerate(U):
        if not inside[n]:
            for k in acc: acc[k].append(-np.inf)
            continue
        l0=lnL0(u,D); acc['l0'].append(lpu[n]-lq[n]+l0)
        acc['P0'].append(lpu[n]-lq[n]+l0+band_lnZ(u,D,4.26838)); acc['al'].append(lpu[n]-lq[n]+l0+band_lnZ(u,D,1.30131))
        acc['P0u'].append(lpu[n]-lq[n]+l0+band_lnZ(u,D,4.26838,True)); acc['alu'].append(lpu[n]-lq[n]+l0+band_lnZ(u,D,1.30131,True))
    Z={k:lse(np.array(v)) for k,v in acc.items()}
    out[name]=dict(lnB_P0=Z['P0']-Z['l0'],lnB_alias=Z['al']-Z['l0'],diff_jeffreys=Z['P0']-Z['al'],diff_uniform_f=Z['P0u']-Z['alu'])
    print(name,out[name],flush=True)
json.dump(out,open('aliasband_marg.json','w'),indent=1)
