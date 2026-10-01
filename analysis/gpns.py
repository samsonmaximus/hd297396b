import numpy as np, pandas as pd, sys, dynesty, pickle, time
from scipy.linalg import cho_factor, cho_solve
from multiprocessing import Pool
G=pd.read_csv('frozen104.csv')
drop=sys.argv[2]=='out' if len(sys.argv)>2 else False
if drop: G=G[np.abs(G.t-2454922.53)>0.2].reset_index(drop=True)
t=G.t.values; tr=t-2455000.0; rv=G.rv.values; erv=G.erv.values; lw=G.lw.values; elw=G.elw.values; li=G.li.values
r=np.abs(tr[:,None]-tr[None,:])
lwmed=np.median(lw)
MODEL=sys.argv[1]  # '0p' or '1p'
npl=3 if MODEL=='1p' else 0
ndim=21+npl
def prior(u):
    x=np.empty(ndim); i=0
    x[0:4]=-50+100*u[0:4]; x[4:8]=10**(-2+np.log10(5000)*u[4:8]); x[8]=10**(-1+2*u[8]); x[9]=10**(-1+np.log10(500)*u[9])
    x[10:14]=lwmed-50+100*u[10:14]; x[14:18]=10**(-2+np.log10(5000)*u[14:18]); x[18]=10**(-1+2*u[18]); x[19]=10**(-1+np.log10(500)*u[19])
    x[20]=10**(np.log10(50)+2*u[20])
    if npl:
        x[21]=4.20*(4.34/4.20)**u[21]; x[22]=30*u[22]; x[23]=u[23]
    return x
def gll(y,e,off,jit,beta,A,ell):
    s3=np.sqrt(3)*r/ell; K=A*A*(1+s3)*np.exp(-s3)
    K[np.diag_indices_from(K)]+= (beta*e)**2+jit[li]**2
    res=y-off[li]
    try: c=cho_factor(K,lower=True,check_finite=False)
    except np.linalg.LinAlgError: return -1e25
    return -0.5*res@cho_solve(c,res,check_finite=False)-np.sum(np.log(np.diag(c[0])))-0.5*len(y)*np.log(2*np.pi)
def loglike(x):
    y=rv.copy()
    if npl: y=y-x[22]*np.sin(2*np.pi*(tr/x[21]+x[23]))
    return gll(y,erv,x[0:4],x[4:8],x[8],x[9],x[20])+gll(lw,elw,x[10:14],x[14:18],x[18],x[19],x[20])
if __name__=='__main__':
    seed=int(sys.argv[3]) if len(sys.argv)>3 else 1
    t0=time.time()
    s=dynesty.NestedSampler(loglike,prior,ndim,nlive=600,sample='rslice',bound='multi',rstate=np.random.default_rng(seed))
    it=0
    for r_ in s.sample(dlogz=0.1):
        it+=1
        if it%2000==0: print('it',it,'ncall',s.ncall,'t',int(time.time()-t0),flush=True)
    s.add_final_live()
    res=s.results
    tag=f'{MODEL}_{"out" if drop else "in"}_s{seed}'
    pickle.dump(dict(logz=res.logz[-1],logzerr=res.logzerr[-1],samples=res.samples,logwt=res.logwt,logl=res.logl),open(f'ns_{tag}.pkl','wb'))
    print(tag,'lnZ=%.3f +- %.3f'%(res.logz[-1],res.logzerr[-1]),'time %.0fs'%(time.time()-t0),'ncall',np.sum(res.ncall))
