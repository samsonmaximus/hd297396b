# Nested sampling of circular and eccentric one-planet models in the white-noise frame (per-label
# offsets and jitters), to test whether an eccentric solution can absorb the discrepant night.
import numpy as np, dynesty, sys, json, pickle
exec(open('ecc.py').read().split('res={}')[0])
which=sys.argv[1]; model=sys.argv[2]; seed=int(sys.argv[3]) if len(sys.argv)>3 else 1
df=S if which=='104' else S[np.abs(S.t-2454922.53)>0.2].reset_index(drop=True)
t=df.t.values; y=df.y.values; e=df.e.values; li=np.array([LABS.index(l) for l in df.lab])
ecc=model=='ecc'; ndim=13 if ecc else 11
T0=2456298.34-4.34/2
def prior(u):
    x=np.empty(ndim)
    x[0]=4.20*(4.34/4.20)**u[0]; x[1]=T0+4.34*u[1]; x[2]=30*u[2]
    k=3
    if ecc:
        ee=0.95*u[3]; w=2*np.pi*u[4]; x[3]=np.sqrt(ee)*np.cos(w); x[4]=np.sqrt(ee)*np.sin(w); k=5
    x[k:k+4]=-50+100*u[k:k+4]; x[k+4:k+8]=np.exp(np.log(0.01)+np.log(5000)*u[k+4:k+8])
    return x
def loglike(x):
    P,Tc,K=x[:3]
    if ecc: sec,ses=x[3],x[4]; k=5
    else: sec=ses=0.0; k=3
    m=kep_rv(t,P,Tc,K,sec,ses)+x[k:k+4][li]; s2=e**2+x[k+4:k+8][li]**2
    return -0.5*np.sum((y-m)**2/s2+np.log(2*np.pi*s2))
def loglike0(x):
    m=x[0:4][li]; s2=e**2+x[4:8][li]**2
    return -0.5*np.sum((y-m)**2/s2+np.log(2*np.pi*s2))
def prior0(u):
    x=np.empty(8); x[0:4]=-50+100*u[0:4]; x[4:8]=np.exp(np.log(0.01)+np.log(5000)*u[4:8]); return x
if model=='0p':
    s=dynesty.NestedSampler(loglike0,prior0,8,nlive=800,sample='rslice',bound='multi',rstate=np.random.default_rng(seed))
else:
    s=dynesty.NestedSampler(loglike,prior,ndim,nlive=1500,sample='rslice',bound='multi',rstate=np.random.default_rng(seed))
s.run_nested(dlogz=0.1,print_progress=False)
r=s.results; out=dict(which=which,model=model,logz=float(r.logz[-1]),logzerr=float(r.logzerr[-1]))
if ecc:
    w=np.exp(r.logwt-r.logz[-1]); smp=r.samples; ee=smp[:,3]**2+smp[:,4]**2
    o=np.argsort(ee); cw=np.cumsum(w[o]); out['e95']=float(ee[o][np.searchsorted(cw,0.95*cw[-1])]); out['e50']=float(ee[o][np.searchsorted(cw,0.5*cw[-1])])
    out['frac_e_gt_0.7']=float(w[ee>0.7].sum()/w.sum())
print(json.dumps(out),flush=True)
json.dump(out,open(f'ecc_ns_{which}_{model}_s{seed}.json','w'))
