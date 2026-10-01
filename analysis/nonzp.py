import numpy as np
exec(open('pg.py').read())
Sraw=nightbin(d[keep],'DRVmlc','e_DRVmlc')
for name,df in [('104',Sraw),('103',Sraw[np.abs(Sraw.t-2454922.53)>0.2])]:
    z,_=periodogram(df,freqs); k=np.argmin(abs(freqs-1/4.26838)); print('no-NZP',name,'dchi2(P0) %.1f  global max %.1f at %.4f'%(z[k-3:k+4].max(),z.max(),1/freqs[np.argmax(z)]))
# 9-sigma: excursion relative to the scatter of the other velocities after offsets
S3=S[np.abs(S.t-2454922.53)>0.2]
jj,_,idx,L,labs=jitter_fit(S3); w=1/(S3.e.values**2+jj[idx]**2); p=np.linalg.solve(L.T@(L*w[:,None]),L.T@(w*S3.y.values)); r=S3.y.values-L@p
i=np.argmin(abs(S.t-2454922.53)); y9=S.y.values[i]-p[labs.index('pre_072')]
print('night minus 072 offset %.1f; rms of other 103 %.2f -> %.1f sigma; 072-only rms %.2f -> %.1f'%(y9,r.std(),y9/r.std(),r[S3.lab.values=='pre_072'].std(),y9/r[S3.lab.values=='pre_072'].std()))
