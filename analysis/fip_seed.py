# Marginalised full-band evidences for 0, 1 and 2 circular planets and the false-inclusion probability
# (FIP; Hara et al. 2022) of the P0 band, in the white-noise frame with every noise parameter marginalised.
# Model: per-label offsets U(-50,50), per-label jitters log-U(0.01,50) m/s, periods Jeffreys(1.05,1000) d,
# K ~ U(0,30) m/s, phase ~ U(0,2pi): the priors of Table A.1 with the period prior widened to the full band.
# Offsets and sinusoid amplitudes are integrated analytically on a grid of 1.24e5 frequencies (Rice form
# for the amplitude prior, see fastcheck.py); the four jitters are integrated by importance sampling.
# The narrow-band (Jeffreys 4.20-4.34 d) one-planet evidence is computed alongside as a check on the
# nested-sampling (juliet) value of Table 3.
import numpy as np, json, sys, time
from scipy.stats import multivariate_t
from scipy.optimize import minimize
exec(open('fastcheck.py').read().split("if __name__")[0])
NJ=int(sys.argv[1]) if len(sys.argv)>1 else 240
MAXC=600; CUT=10.0; FINE=20; LNJ=(np.log(0.01),np.log(50.0))
Tb=S.t.max()-S.t.min(); fr=np.arange(1/1000,1/1.05,1/(FINE*Tb)); dfq=fr[1]-fr[0]
lpri=np.log(dfq/(fr*np.log(1000/1.05))); inband=(fr>=1/4.34)&(fr<=1/4.20)
lpri_n=np.where(inband,np.log(dfq/(fr*np.log(4.34/4.20))),-np.inf)
def setup(df):
    labs=[l for l in LABS if (df.lab==l).any()]; idx=np.array([labs.index(l) for l in df.lab])
    L=np.array([[1.0*(l==x) for x in labs] for l in df.lab]); t=df.t.values; y=df.y.values; e=df.e.values
    ph=2*np.pi*np.outer(t-t.mean(),fr)
    return labs,idx,L,t,y,e
def lnL0(u,D):
    labs,idx,L,t,y,e=D; s2=e**2+np.exp(u)[idx]**2; w=1/s2
    F=L.T@(L*w[:,None]); b=L.T@(w*y); p=np.linalg.solve(F,b); r=y-L@p
    chi=np.sum(w*r*r)
    return -0.5*chi-0.5*np.sum(np.log(2*np.pi*s2))+0.5*len(labs)*np.log(2*np.pi)-0.5*np.linalg.slogdet(F)[1]-len(labs)*np.log(100.0)
def proj(u,D):
    labs,idx,L,t,y,e=D; w=1/np.sqrt(e**2+np.exp(u)[idx]**2)
    Q,_=np.linalg.qr(L*w[:,None]); P=np.eye(len(t))-Q@Q.T; yp=P@(y*w)
    Cm=np.empty((len(fr),len(t))); Sm=np.empty((len(fr),len(t)))
    for i0 in range(0,len(fr),4000):
        ph=2*np.pi*np.outer(t-t.mean(),fr[i0:i0+4000])
        Cm[i0:i0+4000]=(P@(np.cos(ph)*w[:,None])).T; Sm[i0:i0+4000]=(P@(np.sin(ph)*w[:,None])).T
    return Cm,Sm,yp
def evid(u,D,do2=True):
    l0=lnL0(u,D); Cm,Sm,yp=proj(u,D)
    cc=(Cm**2).sum(1); ss=(Sm**2).sum(1); cs=(Cm*Sm).sum(1); a=Cm@yp; b=Sm@yp
    lz1,_=lnZratio_fast(np.c_[cc,ss,cs,a,b])
    w1=lz1+lpri; l1=lse(w1); l1n=lse(lz1[inband]+lpri_n[inband]); l1b=lse(w1[inband])
    out=dict(l0=l0,l1=l0+l1,l1n=l0+l1n,l1band=l0+l1b)
    if not do2: return out
    cand=np.where(lz1>lz1.max()-CUT)[0]
    if len(cand)>MAXC: cand=np.argsort(lz1)[::-1][:MAXC]
    incand=np.zeros(len(fr),bool); incand[cand]=True
    tot_all=[];tot_CC=[];band_all=[];band_CC=[]
    for j0 in range(0,len(cand),128):
        cb=cand[j0:j0+128]; C1=Cm[cb]; S1=Sm[cb]; V=np.vstack([C1,S1]).T; DC=Cm@V; DS=Sm@V; nb=len(cb)
        for k,i in enumerate(cb):
            X1=np.c_[C1[k],S1[k]]; Gi=np.linalg.inv(X1.T@X1)
            dc=np.c_[DC[:,k],DC[:,nb+k]]; ds=np.c_[DS[:,k],DS[:,nb+k]]; dy=X1.T@yp
            cc2=cc-np.einsum('ij,jk,ik->i',dc,Gi,dc); ss2=ss-np.einsum('ij,jk,ik->i',ds,Gi,ds); cs2=cs-np.einsum('ij,jk,ik->i',dc,Gi,ds)
            a2=a-dc@(Gi@dy); b2=b-ds@(Gi@dy)
            ok=(cc2*ss2-cs2**2)>1e-4*(cc*ss)
            lz2=np.full(len(fr),-np.inf); lz2[ok]=lnZratio_fast(np.c_[cc2,ss2,cs2,a2,b2][ok])[0]
            W=w1[i]+lz2+lpri
            tot_all.append(lse(W[ok])); tot_CC.append(lse(W[ok&incand]))
            if inband[i]: band_all.append(tot_all[-1]); band_CC.append(tot_CC[-1])
            else:
                m=ok&inband; band_all.append(lse(W[m]) if m.any() else -np.inf)
                mc=ok&inband&incand; band_CC.append(lse(W[mc]) if mc.any() else -np.inf)
    A=lse(np.array(tot_all)); B=lse(np.array(tot_CC)); l2=A+np.log(2-np.exp(B-A))
    Ab=lse(np.array(band_all)); Bb=lse(np.array(band_CC)); l2b=Ab+np.log(2-np.exp(Bb-Ab))
    out.update(l2=l0+l2,l2band=l0+l2b,ncand=len(cand)); return out
def map_u(D,P=None,P2=None):
    labs,idx,L,t,y,e=D
    cols=[]
    for PP in [P,P2]:
        if PP: cols+= [np.cos(2*np.pi*t/PP),np.sin(2*np.pi*t/PP)]
    X=np.hstack([L,np.array(cols).T]) if cols else L
    def nll(u):
        s2=e**2+np.exp(u)[idx]**2; w=1/s2; F=X.T@(X*w[:,None]); p=np.linalg.solve(F,X.T@(w*y)); r=y-X@p
        return 0.5*np.sum(w*r*r+np.log(s2))+0.5*np.linalg.slogdet(F)[1]
    best=min((minimize(nll,np.full(len(labs),v),method='Nelder-Mead',options=dict(maxiter=4000,xatol=1e-4,fatol=1e-6)) for v in (0.,1.,1.7)),key=lambda r:r.fun)
    # numerical Hessian
    h=1e-3; n=len(best.x); H=np.zeros((n,n)); f0=best.fun
    for i in range(n):
        for j in range(n):
            ei=np.eye(n)[i]*h; ej=np.eye(n)[j]*h
            H[i,j]=(nll(best.x+ei+ej)-nll(best.x+ei-ej)-nll(best.x-ei+ej)+nll(best.x-ei-ej))/(4*h*h)
    return best.x,np.linalg.inv(H+1e-6*np.eye(n))
P0=4.26838
res={}
rng=np.random.default_rng(int(sys.argv[2]) if len(sys.argv)>2 else 11)
for name,df in [('104',S)]:
    D=setup(df); t0=time.time()
    comps=[map_u(D),map_u(D,P0),map_u(D,P0,200.886)]
    qs=[multivariate_t(loc=m,shape=3.0*C,df=4) for m,C in comps]
    U=np.vstack([q.rvs(size=NJ//3,random_state=rng) for q in qs])
    lq=np.log(np.mean([np.exp(q.logpdf(U)) for q in qs],axis=0))
    inside=np.all((U>LNJ[0])&(U<LNJ[1]),axis=1); lpu=np.where(inside,-4*np.log(LNJ[1]-LNJ[0]),-np.inf)
    R=[]
    for n,u in enumerate(U):
        if not inside[n]: R.append(None); continue
        R.append(evid(u,D))
        if n%10==0: print(name,n,round(time.time()-t0),'s',{k:round(v,2) for k,v in R[-1].items()},flush=True)
    keys=['l0','l1','l1n','l1band','l2','l2band']
    lw={k:np.array([lpu[n]-lq[n]+R[n][k] if R[n] else -np.inf for n in range(len(U))]) for k in keys}
    Z={k:lse(lw[k])-np.log(len(U)) for k in keys}
    ess={k:float(np.exp(2*lse(lw[k])-lse(2*lw[k]))) for k in keys}
    lnB1=Z['l1']-Z['l0']; lnB2=Z['l2']-Z['l0']; lnB1n=Z['l1n']-Z['l0']
    post=np.exp(np.array([0,lnB1,lnB2])-lse(np.array([0,lnB1,lnB2])))
    TIP1=np.exp(Z['l1band']-Z['l1']); TIP2=np.exp(Z['l2band']-Z['l2'])
    FIP=1-(post[1]*TIP1+post[2]*TIP2)
    res[name]=dict(lnB1_full=lnB1,lnB2_full=lnB2,lnB1_narrow=lnB1n,post=post.tolist(),TIP1=TIP1,TIP2=TIP2,FIP=FIP,ess=ess,NJ=len(U))
    print(name,json.dumps(res[name]),flush=True)
    json.dump(res,open(f'fip_seed{sys.argv[2]}.json','w'),indent=1)
