import numpy as np
exec(open('semian.py').read().split('res={}')[0])
from scipy.special import i0e
def lnZratio_fast(T5):
    cc,ss,cs,a,b=T5.T; det=cc*ss-cs**2
    dchi=(ss*a*a-2*cs*a*b+cc*b*b)/det
    th=np.c_[(ss*a-cs*b)/det,(cc*b-cs*a)/det]
    Sxx=ss/det; Syy=cc/det; Sxy=-cs/det
    s2=np.sqrt(Sxx*Syy-Sxy**2)
    mu2=th[:,0]**2+th[:,1]**2
    Er=np.sqrt(np.pi/(2*s2))*i0e(mu2/(4*s2))
    return dchi/2-0.5*np.log(det)+np.log(Er)-np.log(KMAX), dchi
if __name__=='__main__':
    Tb=S.t.max()-S.t.min()
    for name,df in [('104',S),('103',S[np.abs(S.t-2454922.53)>0.2])]:
        jj=jitter_fit(df)[0]; fr=np.arange(1/1000,1/1.05,1/(20*Tb)); dfq=fr[1]-fr[0]
        T5=terms(df,fr,jj)
        lz,_=lnZratio(T5); lzf,_=lnZratio_fast(T5)
        lpri=np.log(dfq/(fr*np.log(1000/1.05)))
        cc,ss,cs=T5[:,0],T5[:,1],T5[:,2]; det=cc*ss-cs**2; Sxx=ss/det;Syy=cc/det;Sxy=-cs/det
        tr=Sxx+Syy; dd=np.sqrt((Sxx-Syy)**2/4+Sxy**2); ratio=(tr/2+dd)/(tr/2-dd)
        print(name,'lnB exact %.3f fast %.3f ; max |dlz| %.3f ; median anisotropy %.2f, 99pct %.2f'%(lse(lz+lpri),lse(lzf+lpri),np.abs(lz-lzf).max(),np.median(ratio),np.percentile(ratio,99)))
