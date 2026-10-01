import numpy as np, pandas as pd
exec(open('core.py').read().split("S=nightbin")[0])
def nb2(df):
    df=df.copy(); df['night']=np.floor(df.BJD-0.196).astype(int); out=[]
    for (n,l),g in df.groupby(['night','lab']):
        w=1/g.e_DRVmlcnzp**2; wl=1/g.e_dLW**2
        out.append(dict(t=np.sum(w*g.BJD)/w.sum(), rv=np.sum(w*g.DRVmlcnzp)/w.sum(), erv=1/np.sqrt(w.sum()),
                        lw=np.sum(wl*g.dLW)/wl.sum(), elw=1/np.sqrt(wl.sum()), lab=l, prog=g.ProgID.iloc[0]))
    return pd.DataFrame(out).sort_values('t').reset_index(drop=True)
G=nb2(d[keep])
LABS=['pre_072','pre_183','pre_oth','post']
G['li']=[LABS.index(l) for l in G.lab]
G.to_csv('frozen104.csv',index=False)
print(len(G), G.groupby('lab').size().to_dict(), G.elw.median())
