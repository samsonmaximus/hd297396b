# Two-planet extension of semian.py: evidence of the two-circular-planet model with both periods free
# over the full band, and the false-inclusion probability (Hara et al. 2022) of the P0 band.
# lnZ2(f1,f2)/Z0 = lnZ1(f1) + lnZ(f2 | f1): the second term is the one-planet expression evaluated on
# quantities from which the f1 sinusoid has been projected out (Schur complement). The amplitude
# integral of planet 1 uses its one-planet marginal; cross-terms are negligible for separated periods.
# Amplitude integrals use the isotropic (Rice) form, checked against the exact form in fastcheck.py
# (Delta lnB <= 0.002 on the one-planet problem).
import numpy as np, json, sys
exec(open('fastcheck.py').read().split("if __name__")[0])
FINE=int(sys.argv[1]) if len(sys.argv)>1 else 20
CUT=float(sys.argv[2]) if len(sys.argv)>2 else 12.0
def vecs(df,fr,jj):
    labs=[l for l in LABS if (df.lab==l).any()]; idx=np.array([labs.index(l) for l in df.lab])
    L=np.array([[1.0*(l==x) for x in labs] for l in df.lab])
    t=df.t.values; s=np.sqrt(df.e.values**2+jj[idx]**2); w=1/s
    Q,_=np.linalg.qr(L*w[:,None]); P=np.eye(len(t))-Q@Q.T; yp=P@(df.y.values*w)
    Cm=np.empty((len(fr),len(t))); Sm=np.empty((len(fr),len(t)))
    for i0 in range(0,len(fr),4000):
        ch=fr[i0:i0+4000]; ph=2*np.pi*np.outer(t-t.mean(),ch)
        Cm[i0:i0+4000]=(P@(np.cos(ph)*w[:,None])).T; Sm[i0:i0+4000]=(P@(np.sin(ph)*w[:,None])).T
    return Cm,Sm,yp
Tb=S.t.max()-S.t.min(); res={}
for name,df in [('104',S),('103',S[np.abs(S.t-2454922.53)>0.2])]:
    jj=jitter_fit(df)[0]
    fr=np.arange(1/1000,1/1.05,1/(FINE*Tb)); dfq=fr[1]-fr[0]
    Cm,Sm,yp=vecs(df,fr,jj)
    cc=(Cm**2).sum(1); ss=(Sm**2).sum(1); cs=(Cm*Sm).sum(1); a=Cm@yp; b=Sm@yp
    lz1,dchi1=lnZratio_fast(np.c_[cc,ss,cs,a,b])
    lpri=np.log(dfq/(fr*np.log(1000/1.05)))
    w1=lz1+lpri; lnB1=lse(w1)
    cand=np.where(lz1>lz1.max()-CUT)[0]
    inband=(fr>=1/4.34)&(fr<=1/4.20); incand=np.zeros(len(fr),bool); incand[cand]=True
    print(name,'grid',len(fr),'candidates',len(cand),'lnB1 %.3f'%lnB1,flush=True)
    tot_all=[];tot_CC=[];band_all=[];band_CC=[]; best=(-np.inf,None)
    for j0 in range(0,len(cand),64):
        cb=cand[j0:j0+64]
        C1=Cm[cb]; S1=Sm[cb]
        DC=Cm@np.vstack([C1,S1]).T; DS=Sm@np.vstack([C1,S1]).T   # (nf, 2nb)
        nb=len(cb)
        for k,i in enumerate(cb):
            X1=np.c_[C1[k],S1[k]]; G=X1.T@X1; Gi=np.linalg.inv(G)
            dc=np.c_[DC[:,k],DC[:,nb+k]]; ds=np.c_[DS[:,k],DS[:,nb+k]]; dy=X1.T@yp
            cc2=cc-np.einsum('ij,jk,ik->i',dc,Gi,dc); ss2=ss-np.einsum('ij,jk,ik->i',ds,Gi,ds); cs2=cs-np.einsum('ij,jk,ik->i',dc,Gi,ds)
            a2=a-dc@(Gi@dy); b2=b-ds@(Gi@dy)
            ok=(cc2*ss2-cs2**2)>1e-4*(cc*ss)
            lz2=np.full(len(fr),-np.inf); lz2[ok]=lnZratio_fast(np.c_[cc2,ss2,cs2,a2,b2][ok])[0]
            W=w1[i]+lz2+lpri
            j=np.argmax(W)
            if W[j]>best[0]: best=(W[j],(1/fr[i],1/fr[j]))
            tot_all.append(lse(W[ok])); tot_CC.append(lse(W[ok&incand]))
            if inband[i]:
                band_all.append(tot_all[-1]); band_CC.append(tot_CC[-1])
            else:
                m=ok&inband; band_all.append(lse(W[m]) if m.any() else -np.inf)
                mc=ok&inband&incand; band_CC.append(lse(W[mc]) if mc.any() else -np.inf)
    A=lse(np.array(tot_all)); Bc=lse(np.array(tot_CC))
    lnZ2=A+np.log(2-np.exp(Bc-A))
    Ab=lse(np.array(band_all)); Bb=lse(np.array(band_CC))
    lnZ2b=Ab+np.log(2-np.exp(Bb-Ab))
    lnB2=lnZ2-np.log(1.0)   # prior over (f1,f2) is the normalised product; pairs counted in both orders
    lw=np.array([0.0,lnB1,lnB2]); post=np.exp(lw-lse(lw))
    TIP1=np.exp(lse(w1[inband])-lnB1); TIP2=np.exp(lnZ2b-lnZ2)
    FIP=1-(post[1]*TIP1+post[2]*TIP2)
    res[name]=dict(lnB1=float(lnB1),lnB2=float(lnB2),lnB21=float(lnB2-lnB1),post=post.tolist(),TIP1=float(TIP1),TIP2=float(TIP2),FIP=float(FIP),best_pair=best[1],ncand=int(len(cand)),fine=FINE,cut=CUT)
    print(name,res[name],flush=True)
json.dump(res,open(f'semian2_{FINE}_{int(CUT)}.json','w'),indent=1)
