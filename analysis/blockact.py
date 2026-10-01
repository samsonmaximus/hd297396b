# Does the P0 amplitude track the activity level? Four equal-length blocks (as Sect. 5.2):
# block amplitude at P0 vs block-mean log R'HK and dLW.
import numpy as np
exec(open('split.py').read().split('subsets={')[0])
from scipy.stats import pearsonr, spearmanr
df=S; t0,t1=df.t.min(),df.t.max(); edges=np.linspace(t0,t1+1e-6,5)
jn=jitter_fit(df)[0]; labs_all=[l for l in LABS if (df.lab==l).any()]
rows=[]
for k in range(4):
    b=df[(df.t>=edges[k])&(df.t<edges[k+1])]
    labs=[l for l in LABS if (b.lab==l).any()]; idx=np.array([labs.index(l) for l in b.lab])
    jj=np.array([jn[labs_all.index(l)] for l in labs])
    s2=b.e.values**2+jj[idx]**2; w=1/s2
    L=np.array([[1.0*(l==x) for x in labs] for l in b.lab]); X=np.hstack([L,np.c_[np.cos(2*np.pi*b.t/P0),np.sin(2*np.pi*b.t/P0)]])
    A=X.T@(X*w[:,None]); p=np.linalg.solve(A,X.T@(w*b.y.values)); C=np.linalg.inv(A); K=np.hypot(*p[-2:])
    J=p[-2:]/K; sK=np.sqrt(J@C[-2:,-2:]@J)
    rhk=np.log10(np.nanmean(b.rhk)); lw=np.nanmean(b.dLW)
    rows.append((k+1,len(b),K,sK,rhk,lw))
    print('block %d n=%d K=%.2f±%.2f  <logR\'HK>=%.3f  <dLW>=%.1f'%rows[-1])
R=np.array(rows)
# weighted slope of K on logR'HK
x=R[:,4]; y=R[:,2]; s=R[:,3]; w=1/s**2
A=np.c_[np.ones(4),x-x.mean()]; C=np.linalg.inv(A.T@(A*w[:,None])); p=C@A.T@(w*y)
print('slope dK/dlogRHK = %.1f ± %.1f m/s per dex ; spread of block logRHK = %.3f dex'%(p[1],np.sqrt(C[1,1]),x.max()-x.min()))
print('chi2 constant K: %.2f (3 dof)'%np.sum(w*(y-np.sum(w*y)/w.sum())**2))
