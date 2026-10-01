import numpy as np, pandas as pd
from scipy.optimize import minimize
d=pd.read_csv('data/HD297396_rvbank_full.csv')
FIB=2457170.0
def label(r):
    if r.BJD>FIB: return 'post'
    if r.ProgID.startswith('072.C-0488'): return 'pre_072'
    if r.ProgID.startswith('183.C-0972'): return 'pre_183'
    return 'pre_oth'
d['lab']=d.apply(label,axis=1)
# criteria
d['C2']=False
for era,g in d.groupby(d.BJD>FIB):
    split=g.DRVmlcnzp-(g.RVdrsnzp-np.median(g.RVdrsnzp))
    split=split-np.median(split)
    s=np.median(np.abs(split-np.median(split)))*1.4826
    d.loc[g.index,'C2']=np.abs(split)>5*s
d['C3']=np.abs(d.DRIFT)>3.0
d['C4']=d.SNRDRS<20
keep=~(d.C2|d.C3|d.C4|(d.Flag!=0))
print('rejected C2',d.C2.sum(),'C3',d.C3.sum(),'C4',d.C4.sum(),'kept',keep.sum())
def nightbin(df,rv,erv):
    df=df.copy(); df['night']=np.floor(df.BJD-0.196).astype(int)  # nights from La Silla local noon (16:43 UT = JD fraction 0.196)
    out=[]
    for (n,l),g in df.groupby(['night','lab']):
        w=1/g[erv]**2
        out.append(dict(t=np.sum(w*g.BJD)/w.sum(), y=np.sum(w*g[rv])/w.sum(), e=1/np.sqrt(w.sum()), lab=l, prog=g.ProgID.iloc[0], n=len(g),
                        **{k:np.sum(w*g[k])/w.sum() for k in ['dLW','Halpha','NaD1','NaD2','FWHMDRS','Contrast','BIS','CRX']},
                        rhk=np.nanmean(g.RHKp)))
    return pd.DataFrame(out).sort_values('t').reset_index(drop=True)
S=nightbin(d[keep],'DRVmlcnzp','e_DRVmlcnzp')
D=nightbin(d[keep],'RVdrsnzp','e_RVdrsnzp')
print('epochs',len(S), S.lab.value_counts().to_dict(), 'median err',S.e.median())
LABS=['pre_072','pre_183','pre_oth','post']
