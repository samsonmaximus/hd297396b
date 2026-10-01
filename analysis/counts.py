import gzip, numpy as np, pandas as pd
rows=[]
with gzip.open('data/table4.dat.gz','rt') as f:
    for l in f:
        try: rows.append((l[0:14].strip(), float(l[58:73]), float(l[74:87]), float(l[88:98])))
        except: rows.append((l[0:14].strip(), float(l[58:73]) if l[58:73].strip() else np.nan, np.nan, np.nan))
T=pd.DataFrame(rows,columns=['star','bjd','rv','e'])
print('rows',len(T),'stars',T.star.nunique())
T=T[np.isfinite(T.rv)]
T['night']=np.floor(T.bjd-0.196)
g=T.groupby('star').agg(nrv=('rv','size'),nn=('night','nunique'),base=('bjd',lambda x:x.max()-x.min()),me=('e','median'))
print('>=20 RVs & base>100 d:',((g.nrv>=20)&(g.base>100)).sum())
print('>=20 nights & base>100 d:',((g.nn>=20)&(g.base>100)).sum())
print('>=20 nights & base>=365 & med err<=5:',((g.nn>=20)&(g.base>=365)&(g.me<=5)).sum())
print('HD297396:',g.loc['HD297396'].to_dict())
