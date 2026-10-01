# v15: TOI-6263.01 radius, predicted semi-amplitude and Hill separation from the ExoFOP depth
# (193 +- 17 ppm; QLP, S/N 9) and the stellar parameters of Table 2.
import numpy as np, json
rng = np.random.default_rng(6263); N = 200000
RSUN_REARTH = 109.076; MEARTH_MSUN = 3.0035e-6; MJUP_MEARTH = 317.83
dep = rng.normal(193e-6, 17e-6, N); Rs = rng.normal(0.739, 0.026, N); Ms = rng.normal(0.776, 0.047, N)
Rp = Rs*RSUN_REARTH*np.sqrt(dep)                        # no limb-darkening or impact-parameter correction
# Chen & Kipping (2017) terran branch R = M^0.279 (M < 2.04 M_E); invert, with 0.04-dex intrinsic scatter in R
Mp = (Rp*10**rng.normal(0, 0.0403, N))**(1/0.279)
Pc, Pb = 3.0178746, 4.26837
def Kpred(Mp_e, P_d, Ms_sun):                            # circular, sin i = 1
    return 28.4329*(Mp_e/MJUP_MEARTH)*Ms_sun**(-2/3)*(P_d/365.25)**(-1/3)
Kc = Kpred(Mp, Pc, Ms)
ac = (Ms*(Pc/365.25)**2)**(1/3); ab = (Ms*(Pb/365.25)**2)**(1/3)
mb = rng.normal(11.8, 1.9, N)                           # coplanar with a transiting c: true mass ~ M sin i
RH = ((mb + Mp)*MEARTH_MSUN/(3*Ms))**(1/3)*(ab + ac)/2
Delta = (ab - ac)/RH
q = lambda x: [float(np.percentile(x, s)) for s in (16, 50, 84)]
out = dict(Rp=q(Rp), Mp=q(Mp), Kc=q(Kc), ac=q(ac), Delta=q(Delta), frac_above_gladman=float(np.mean(Delta > 2*np.sqrt(3))),
           Rp_TOI_catalogue=0.931, Rstar_implied_by_catalogue=float(0.931/RSUN_REARTH/np.sqrt(193e-6)))
for k, v in out.items(): print(k, np.round(v, 4) if isinstance(v, list) else v)
json.dump(out, open('toi.json', 'w'), indent=1)
