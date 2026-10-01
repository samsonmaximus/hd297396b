# Baluev (2008) analytic FAP for the per-label-offset, fixed-jitter chi^2 periodogram.
# The statistic is Delta chi^2 (known-variance "psd" normalisation, z = Delta chi^2 / 2).
# tau(z) = W e^{-z} sqrt(z), W = f_max * sqrt(4 pi Var_w(t)), with Var_w the inverse-variance
# weighted variance of the times (Baluev 2008, eq. 14 for the known-noise case; astropy's
# implementation for normalization='psd').  FAP ~ 1 - exp(-tau) for small tau.
import numpy as np
exec(open('pg.py').read())
def baluev(df, zobs, fmin=1/1000, fmax=1/1.05):
    jj,_,idx,_,_ = jitter_fit(df)
    s2 = df.e.values**2 + jj[idx]**2; w = 1/s2
    t = df.t.values
    varw = np.sum(w*t**2)/w.sum() - (np.sum(w*t)/w.sum())**2
    W = (fmax-fmin)*np.sqrt(4*np.pi*varw)
    z = zobs/2
    tau = W*np.exp(-z)*np.sqrt(z)
    return 1-np.exp(-tau), W, np.sqrt(varw)
S103 = S[np.abs(S.t-2454922.53)>0.2]
for name, df in [('104',S),('103',S103)]:
    z,_ = periodogram(df,freqs)
    fap,W,sd = baluev(df, z.max())
    print(name, 'zmax %.2f  W %.0f  sd_t %.0f d  Baluev FAP %.2e  x1221 = %.3g' % (z.max(), W, sd, fap, fap*1221))
    # also for the held-out window-free check: per-trial FAP at fixed frequency
    print('   single-frequency p (chi2_2):', np.exp(-z.max()/2))
