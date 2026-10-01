exec(open('pg.py').read())
import warnings; warnings.filterwarnings('ignore')
P0=4.26838
def fitK(df,P=P0):
    jj,_,idx,L,labs=jitter_fit(df, design=np.c_[np.cos(2*np.pi*df.t/P),np.sin(2*np.pi*df.t/P)])
    s2=df.e.values**2+jj[idx]**2; w=1/s2
    X=np.hstack([L,np.c_[np.cos(2*np.pi*df.t/P),np.sin(2*np.pi*df.t/P)]])
    A=X.T@(X*w[:,None]); p=np.linalg.solve(A,X.T@(w*df.y.values)); C=np.linalg.inv(A)
    a,b=p[-2:]; K=np.hypot(a,b); ph=np.degrees(np.arctan2(b,a))
    J=np.array([a/K,b/K]); sK=np.sqrt(J@C[-2:,-2:]@J)
    Jp=np.array([-b,a])/K**2; sph=np.degrees(np.sqrt(Jp@C[-2:,-2:]@Jp))
    return K,sK,ph,sph,jj
S103=S[np.abs(S.t-2454922.53)>0.2]
subsets={
 'ALL 104':S,'ALL 103':S103,
 'GTO 072.C-0488 only (47)':S[S.lab=='pre_072'],
 'GTO 072 minus 9sig epoch (46)':S103[S103.lab=='pre_072'],
 'Everything except 072.C-0488 (57)':S[S.lab!='pre_072'],
 'Lo Curto continuation progs only':S[S.prog.str.match(r'^(08[5-9]|09\d|010[0-2])')],
 'pre-2015 fibre only (90)':S[S.lab!='post'],
 'post-2015 fibre only (14)':S[S.lab=='post'],
 'first half of time (t<2455000)':S103[S103.t<2455000],
 'second half (t>=2455000)':S103[S103.t>=2455000],
}
for name,df in subsets.items():
    z,jj=periodogram(df,freqs)
    k=np.argmin(abs(freqs-1/P0)); zP=z[k-3:k+4].max()
    rank=(z>zP).sum()
    # count of independent peaks higher: approximate by fraction of grid above
    frac=(z>=zP).mean()
    K,sK,ph,sph,_=fitK(df)
    print(f'{name:38s} n={len(df):3d} dchi2(P0)={zP:6.2f} global-max P={1/freqs[np.argmax(z)]:9.4f} ({z.max():6.2f}) frac_grid_above={frac:.4f}  K={K:5.2f}±{sK:4.2f} phase={ph:7.1f}±{sph:5.1f}')
