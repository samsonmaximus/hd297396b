# Semi-analytic Bayes factors over the full search band, and the posterior distribution of the period.
# Frame: per-label offsets (flat priors, marginalised analytically) and per-label jitters fixed at their
# planet-free maximum-likelihood values (the frame of the periodogram). Priors as Table A.1:
# P ~ Jeffreys, K ~ U(0,30) m/s, phase ~ U(0,2pi).  For each trial frequency the amplitude integral is
#   Z1(f)/Z0 = exp(dchi2/2) * |S|^{-1/2} E_{N(theta_hat, S^-1)}[1/|theta|] / Kmax,
# with S the offset-projected Fisher matrix of (a,b); E[1/|theta|] is evaluated exactly by the identity
# 1/r = pi^{-1/2} int_0^inf s^{-1/2} exp(-s r^2) ds.
import numpy as np, json
exec(open('pg.py').read())
KMAX=30.0
def terms(df,fr,jj):
    labs=[l for l in LABS if (df.lab==l).any()]; idx=np.array([labs.index(l) for l in df.lab])
    L=np.array([[1.0*(l==x) for x in labs] for l in df.lab])
    t=df.t.values; s=np.sqrt(df.e.values**2+jj[idx]**2); w=1/s
    Q,_=np.linalg.qr(L*w[:,None]); P=np.eye(len(t))-Q@Q.T; yp=P@(df.y.values*w)
    out=[]
    for ch in np.array_split(fr,max(1,len(fr)//3000)):
        ph=2*np.pi*np.outer(t-t.mean(),ch)
        C=P@(np.cos(ph)*w[:,None]); Sn=P@(np.sin(ph)*w[:,None])
        out.append(np.c_[(C*C).sum(0),(Sn*Sn).sum(0),(C*Sn).sum(0),C.T@yp,Sn.T@yp])
    return np.vstack(out)
u=np.linspace(-14,8,400); sg=np.exp(u); du=u[1]-u[0]
def lnZratio(T5):
    cc,ss,cs,a,b=T5.T; det=cc*ss-cs**2
    dchi=(ss*a*a-2*cs*a*b+cc*b*b)/det
    th=np.c_[(ss*a-cs*b)/det,(cc*b-cs*a)/det]          # LS amplitudes
    # covariance Sigma = S^-1 ; eigen-decomposition of 2x2
    Sxx=ss/det; Syy=cc/det; Sxy=-cs/det
    tr=Sxx+Syy; dd=np.sqrt((Sxx-Syy)**2/4+Sxy**2); l1=tr/2+dd; l2=tr/2-dd
    ang=0.5*np.arctan2(2*Sxy,Sxx-Syy); c_,s_=np.cos(ang),np.sin(ang)
    m1=c_*th[:,0]+s_*th[:,1]; m2=-s_*th[:,0]+c_*th[:,1]
    Er=np.empty(len(cc))
    for i0 in range(0,len(cc),5000):
        sl=slice(i0,i0+5000); s=sg[None,:]
        integ=np.sqrt(s)*((1+2*s*l1[sl,None])*(1+2*s*l2[sl,None]))**-0.5*np.exp(-s*(m1[sl,None]**2/(1+2*s*l1[sl,None])+m2[sl,None]**2/(1+2*s*l2[sl,None])))
        Er[sl]=integ.sum(1)*du/np.sqrt(np.pi)   # ds = s du ; s^{-1/2} * s = s^{1/2}
    return dchi/2+0.5*np.log(det)*-1+np.log(Er)-np.log(KMAX), dchi
def lse(x): m=x.max(); return m+np.log(np.exp(x-m).sum())
res={}
Tb=S.t.max()-S.t.min()
for name,df in [('104',S),('103',S[np.abs(S.t-2454922.53)>0.2])]:
    jj=jitter_fit(df)[0]
    out={}
    for fine in [40,80]:
        fr=np.arange(1/1000,1/1.05,1/(fine*Tb)); dfq=fr[1]-fr[0]
        lz,dchi=lnZratio(terms(df,fr,jj))
        lw_full=lz+np.log(dfq/(fr*np.log(1000/1.05)))
        nb=(fr>=1/4.34)&(fr<=1/4.20)
        lw_nar=lz[nb]+np.log(dfq/(fr[nb]*np.log(4.34/4.20)))
        lnB_full=lse(lw_full); lnB_nar=lse(lw_nar)
        post=np.exp(lw_full-lnB_full)
        def mass(P1,P2): m=(fr>=1/P2)&(fr<=1/P1); return float(post[m].sum())
        out[fine]=dict(lnB_full=float(lnB_full),lnB_narrow=float(lnB_nar),occam=float(lnB_nar-lnB_full),
                       TIP_P0=mass(4.20,4.34),m_alias1301=mass(1.29,1.31),m_200=mass(190,212),
                       m_top=[(float(1/fr[i]),float(post[i])) for i in np.argsort(post)[::-1][:1]])
        # contributions by region
        order=np.argsort(post)[::-1]
        print(name,fine,{k:(round(v,4) if isinstance(v,float) else v) for k,v in out[fine].items()},flush=True)
    res[name]=out[80]
    # posterior probability of 1p vs 0p, equal prior odds, and false inclusion probability of the P0 band
    B=np.exp(res[name]['lnB_full']); p1=B/(1+B)
    res[name]['p_1p']=p1; res[name]['FIP_P0_1p']=1-p1*res[name]['TIP_P0']
    print(name,'P(1p|d)=%.4f  FIP(P0 band | 0p/1p)=%.2e'%(p1,res[name]['FIP_P0_1p']))
json.dump(res,open('semian.json','w'),indent=1)
