# Covariance-weighted (GLS) periodogram under the adopted noise model at its maximum-likelihood
# hyperparameters, and its global FAP from draws of the same covariance.
import numpy as np, pandas as pd, json
exec(open('gpmap.py').read().split("if __name__")[0])
H=json.load(open('gpmap.json'))
rng=np.random.default_rng(5)
def cov(drop,h):
    g,t,r,li,Lm=prep(drop); s3=np.sqrt(3)*r/h['ell']; K=h['Ar']**2*(1+s3)*np.exp(-s3)
    K[np.diag_indices_from(K)]+=(h['br']*g.erv.values)**2+np.array(h['jr'])[li]**2
    return g,t,li,Lm,K
T=6536.0; freqs=np.arange(1/1000,1/1.05,1/(5*T))
def gls_pg(t,Lm,Lc,Y,freqs):
    # whiten with Cholesky factor Lc (lower): Xw = Lc^-1 X
    from scipy.linalg import solve_triangular
    Lw=solve_triangular(Lc,Lm,lower=True); Q,_=np.linalg.qr(Lw); P=np.eye(len(t))-Q@Q.T
    Yw=solve_triangular(Lc,Y,lower=True); yp=P@Yw
    out=[]
    for ch in np.array_split(freqs,max(1,len(freqs)//2000)):
        ph=2*np.pi*np.outer(t,ch)
        C=P@solve_triangular(Lc,np.cos(ph),lower=True); S_=P@solve_triangular(Lc,np.sin(ph),lower=True)
        cc=(C*C).sum(0); ss=(S_*S_).sum(0); cs=(C*S_).sum(0); det=cc*ss-cs**2
        a=C.T@yp; b=S_.T@yp
        if yp.ndim==1: out.append((ss*a*a-2*cs*a*b+cc*b*b)/det)
        else: out.append((ss[:,None]*a*a-2*cs[:,None]*a*b+cc[:,None]*b*b)/det[:,None])
    return np.concatenate(out)
res={}
for drop in [False,True]:
    tag='103' if drop else '104'; h=H[tag+'_0p']
    g,t,li,Lm,K=cov(drop,h); Lc=np.linalg.cholesky(K)
    z=gls_pg(t,Lm,Lc,g.rv.values,freqs); k0=np.argmin(abs(freqs-1/4.26838))
    zP=z[k0-3:k0+4].max(); Pmax=1/freqs[np.argmax(z)]
    # null draws from the same covariance: whitened draws are unit normal
    mx=[]
    for b in range(10):
        Y=Lc@rng.normal(size=(len(t),1000)); mx.append(gls_pg(t,Lm,Lc,Y,freqs).max(0))
    mx=np.concatenate(mx)
    res[tag]=dict(A_rv=h['Ar'],ell=h['ell'],jit=h['jr'],z_P0=float(zP),P_max=float(Pmax),z_max=float(z.max()),
                  n_exceed=int((mx>=z.max()).sum()),ndraw=len(mx),q99=float(np.percentile(mx,99)))
    print(tag,res[tag],flush=True)
json.dump(res,open('gpfap.json','w'),indent=1)
