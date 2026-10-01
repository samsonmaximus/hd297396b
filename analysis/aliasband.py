# Alias comparison with prior-neutral bands: Jeffreys priors of equal logarithmic width centred on
# P0 = 4.268 d and on its sidereal-day alias 1.3013 d (4.20-4.34 d is +-1.65% in log period).
import numpy as np, json
exec(open('fastcheck.py').read().split("if __name__")[0])
Tb=S.t.max()-S.t.min(); out={}
hw=0.5*np.log(4.34/4.20)
for name,df in [('104',S),('103',S[np.abs(S.t-2454922.53)>0.2])]:
    jj=jitter_fit(df)[0]; r={}
    for lab,Pc in [('P0',4.26838),('alias',1.30131)]:
        P1,P2=Pc*np.exp(-hw),Pc*np.exp(hw)
        fr=np.linspace(1/P2,1/P1,20001); dfq=fr[1]-fr[0]
        lz,_=lnZratio_fast(terms(df,fr,jj))
        r[lab]=float(lse(lz+np.log(dfq/(fr*2*hw))))
    out[name]=dict(lnB_P0=r['P0'],lnB_alias=r['alias'],diff=r['P0']-r['alias'])
    print(name,out[name])
json.dump(out,open('aliasband.json','w'),indent=1)
