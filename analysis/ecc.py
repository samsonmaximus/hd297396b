# Eccentricity constraint for HD 297396 b: MCMC (emcee) of one eccentric Keplerian with per-label
# offsets and jitters (white-noise frame), priors as Table A.1 plus e ~ U(0,1) via sqrt(e)cos w, sqrt(e)sin w.
import numpy as np, emcee, json
exec(open('split.py').read().split('subsets={')[0])
def kep_rv(t,P,Tc,K,sec,ses):
    e=sec**2+ses**2; w=np.arctan2(ses,sec)
    # time of periastron from time of conjunction
    f_c=np.pi/2-w; E_c=2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(f_c/2)); M_c=E_c-e*np.sin(E_c)
    tp=Tc-P*M_c/(2*np.pi)
    M=2*np.pi*(t-tp)/P; E=M.copy()
    for _ in range(40): E=E-(E-e*np.sin(E)-M)/(1-e*np.cos(E))
    f=2*np.arctan2(np.sqrt(1+e)*np.sin(E/2),np.sqrt(1-e)*np.cos(E/2))
    return K*(np.cos(f+w)+e*np.cos(w))
res={}
for name,df in [('104',S),('103',S[np.abs(S.t-2454922.53)>0.2].reset_index(drop=True))]:
    t=df.t.values; y=df.y.values; e=df.e.values; li=np.array([LABS.index(l) for l in df.lab])
    def lnp(x):
        P,Tc,K,sec,ses=x[:5]; off=x[5:9]; lj=x[9:13]
        if not (4.20<P<4.34 and 0<K<30 and sec**2+ses**2<0.95 and np.all(np.abs(off)<50) and np.all((lj>np.log(0.01))&(lj<np.log(50))) and 2456290<Tc<2456290+P+10): return -np.inf
        m=kep_rv(t,P,Tc,K,sec,ses)+off[li]; s2=e**2+np.exp(lj)[li]**2
        return -0.5*np.sum((y-m)**2/s2+np.log(s2))-np.log(P)
    rng=np.random.default_rng(3); nw=64
    x0=np.r_[4.26837,2456298.34,5.5,0.0,0.0,np.zeros(4),np.log([5,5,4,4])]
    p0=x0+rng.normal(size=(nw,13))*np.r_[1e-4,0.05,0.3,0.05,0.05,0.5,0.5,0.5,0.5,0.1,0.1,0.1,0.1]
    sam=emcee.EnsembleSampler(nw,13,lnp)
    sam.run_mcmc(p0,24000,progress=False)
    ch=sam.get_chain(discard=8000,thin=10,flat=True)
    ecc=ch[:,3]**2+ch[:,4]**2
    try: tau=sam.get_autocorr_time(discard=8000,quiet=True)
    except Exception as ex: tau=np.array([np.nan])
    res[name]=dict(e95=float(np.percentile(ecc,95)),e_med=float(np.median(ecc)),K=[float(np.median(ch[:,2])),float(np.std(ch[:,2]))],tau_max=float(np.nanmax(tau)),nsamp=len(ch))
    print(name,res[name],flush=True)
json.dump(res,open('ecc.json','w'),indent=1)
