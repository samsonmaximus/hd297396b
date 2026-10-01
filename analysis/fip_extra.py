# Checks on the FIP computation (fip.py), same frame, priors and importance-sampling proposal:
# (a) is a third planet supported once P0 and the 200.9-d signal are in? Z3/Z2 with the first two
#     frequencies fixed at their best values (amplitudes integrated), the third free over 1.05-1000 d;
# (b) how much of the one-planet posterior mass in the 4.20-4.34 d band lies within one resolution
#     element (1/T) of P0, i.e. is the band result the P0 peak or its yearly-alias sidebands?
import numpy as np, json
src=open('fip.py').read()
exec(src.split("P0=4.26838\nres={}")[0].replace("NJ=int(sys.argv[1]) if len(sys.argv)>1 else 240","NJ=240"))
P0=4.26838; P2=200.58; f0=1/P0; Tb=S.t.max()-S.t.min()
core=np.abs(fr-f0)<=1/Tb
def extra(u,D):
    labs,idx,L,t,y,e=D; w=1/np.sqrt(e**2+np.exp(u)[idx]**2)
    X=np.c_[L,np.cos(2*np.pi*t/P0),np.sin(2*np.pi*t/P0),np.cos(2*np.pi*t/P2),np.sin(2*np.pi*t/P2)]
    Xw=X*w[:,None]; yw=y*w
    # log L with offsets integrated (flat, width 100) and both sinusoids' amplitudes integrated (Rice form)
    Q,_=np.linalg.qr(L*w[:,None]); Pm=np.eye(len(t))-Q@Q.T
    A=Pm@Xw[:,4:]; yp=Pm@yw
    l0=lnL0(u,D)
    # two fixed sinusoids: sequential conditioning, as in fip.py
    def lz_of(C,Sn,yv):
        return lnZratio_fast(np.c_[[C@C],[Sn@Sn],[C@Sn],[C@yv],[Sn@yv]])[0][0]
    c1,s1,c2,s2=A.T
    lz1=lz_of(c1,s1,yp)
    X1=np.c_[c1,s1]; G1=np.linalg.inv(X1.T@X1); Pr1=np.eye(len(t))-X1@G1@X1.T
    lz2=lz_of(Pr1@c2,Pr1@s2,Pr1@yp)
    X12=np.c_[c1,s1,c2,s2]; G=np.linalg.inv(X12.T@X12); Pr=np.eye(len(t))-X12@G@X12.T
    # third frequency over the grid
    out=[]
    for i0 in range(0,len(fr),4000):
        ph=2*np.pi*np.outer(t-t.mean(),fr[i0:i0+4000])
        C=Pr@(Pm@(np.cos(ph)*w[:,None])); Sn=Pr@(Pm@(np.sin(ph)*w[:,None])); yv=Pr@yp
        ok=((C*C).sum(0)*(Sn*Sn).sum(0)-((C*Sn).sum(0))**2)>1e-4*(C*C).sum(0)*(Sn*Sn).sum(0)
        z=np.full(C.shape[1],-np.inf)
        z[ok]=lnZratio_fast(np.c_[(C*C).sum(0),(Sn*Sn).sum(0),(C*Sn).sum(0),C.T@yv,Sn.T@yv][ok])[0]
        out.append(z)
    lz3=np.concatenate(out)
    # (b) one-planet band mass near P0
    Cm=None
    ph0=None
    l3=lse(lz3+lpri)
    # one-planet posterior within band vs core window
    labs2=labs
    return l0, l0+lz1+lz2, l0+lz1+lz2+l3
def onep_core(u,D):
    labs,idx,L,t,y,e=D; w=1/np.sqrt(e**2+np.exp(u)[idx]**2)
    Q,_=np.linalg.qr(L*w[:,None]); Pm=np.eye(len(t))-Q@Q.T; yp=Pm@(y*w)
    m=(fr>=1/4.34)&(fr<=1/4.20); f=fr[m]; ph=2*np.pi*np.outer(t-t.mean(),f)
    C=Pm@(np.cos(ph)*w[:,None]); Sn=Pm@(np.sin(ph)*w[:,None])
    lz=lnZratio_fast(np.c_[(C*C).sum(0),(Sn*Sn).sum(0),(C*Sn).sum(0),C.T@yp,Sn.T@yp])[0]+lpri[m]
    return lse(lz), lse(lz[np.abs(f-f0)<=1/Tb])
rng=np.random.default_rng(11); res={}
for name,df in [('103',S[np.abs(S.t-2454922.53)>0.2].reset_index(drop=True)),('104',S)]:
    D=setup(df)
    comps=[map_u(D),map_u(D,P0),map_u(D,P0,200.886)]
    qs=[multivariate_t(loc=m,shape=3.0*C,df=4) for m,C in comps]
    U=np.vstack([q.rvs(size=NJ//3,random_state=rng) for q in qs])
    lq=np.log(np.mean([np.exp(q.logpdf(U)) for q in qs],axis=0))
    inside=np.all((U>LNJ[0])&(U<LNJ[1]),axis=1); lpu=np.where(inside,-4*np.log(LNJ[1]-LNJ[0]),-np.inf)
    acc={k:[] for k in ['z2fix','z3fix','band','core']}
    for n,u in enumerate(U):
        if not inside[n]:
            for k in acc: acc[k].append(-np.inf)
            continue
        l0,l2f,l3f=extra(u,D); b,c=onep_core(u,D); wgt=lpu[n]-lq[n]
        acc['z2fix'].append(wgt+l2f); acc['z3fix'].append(wgt+l3f); acc['band'].append(wgt+l0+b); acc['core'].append(wgt+l0+c)
    Z={k:lse(np.array(v)) for k,v in acc.items()}
    res[name]=dict(lnB32=Z['z3fix']-Z['z2fix'],frac_band_in_core_1p=float(np.exp(Z['core']-Z['band'])))
    print(name,res[name],flush=True)
json.dump(res,open('fip_extra.json','w'),indent=1)
