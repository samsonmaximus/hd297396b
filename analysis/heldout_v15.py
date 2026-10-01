# v15: the out-of-sample test of Sect. 8.3, run on 2026-09-29.
#
# The three HARPS spectra taken after the HARPS-RVBank release were retrieved from the
# ESO Science Archive on 2026-09-29 (Phase 3 products and their ancillary DRS tarballs).
# The values below are copied from the header of each *_ccf_K5_A.fits file inside the
# ancillary tarball (DRS HARPS_3.8, mask K5, DPR TYPE STAR,SKY,K0, no simultaneous drift).
#
# The decision rule is the one frozen in heldout_new.py (v13, 2026-09-25), unchanged:
# D = RV(2024 Mar) - mean(RV(2022 Mar), RV(2024 Nov)); prediction from Table 4 by Monte
# Carlo; noise per epoch = 3.9 m/s residual scatter (+) the DRS DVRMS of each spectrum;
# |z| < 2 consistent with the planet; z0 against D = 0.
#
# Everything after the line "SECONDARY" was NOT pre-registered and is labelled as such.
import json
import numpy as np

HDR = [  # ADP dataset, ancillary tar, BJD (DRS), RVC km/s, DVRMS m/s, CCF NOISE km/s, SN50, SN60, prog, seeing, airmass
    dict(adp='ADP.2022-03-19T01:05:23.555', tar='ADP.2022-03-19T01:05:23.556', date='2022-03-18T04:49:44.956',
         bjd=2459656.70912233, rvc=16.3887741837208, dvrms=3.50267532947247, noise=0.00134414346866798,
         sn50=53.5, sn60=57.5, prog='108.222V.001', seeing=1.25, airm=1.221, fwhm=6.29892077999771, contrast=50.6053257336387),
    dict(adp='ADP.2024-03-21T01:00:35.637', tar='ADP.2024-03-21T01:00:35.638', date='2024-03-20T03:34:26.780',
         bjd=2460389.65739768, rvc=16.3748052278186, dvrms=5.70664868449154, noise=0.00355877052975464,
         sn50=25.5, sn60=25.1, prog='106.21R4.001', seeing=1.26, airm=1.112, fwhm=6.17526099629617, contrast=51.5565283521947),
    dict(adp='ADP.2024-11-16T01:02:11.126', tar='ADP.2024-11-16T01:02:11.127', date='2024-11-15T08:35:35.300',
         bjd=2460629.86179824, rvc=16.3862480181897, dvrms=3.92067502572394, noise=0.00161588609035832,
         sn50=45.6, sn60=43.9, prog='106.21R4.001', seeing=1.26, airm=1.136, fwhm=6.2001289677174, contrast=51.4167417781956),
]

# ---------------------------------------------------------------- frozen criteria
def moon_check(date):
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation, get_body, get_sun, AltAz
    import astropy.units as u
    loc = EarthLocation(lat=-29.2567 * u.deg, lon=-70.7377 * u.deg, height=2400 * u.m)
    t = Time(date, scale='utc', location=loc)
    star = SkyCoord('09h16m47.93s', '-49d18m03.0s')
    moon = get_body('moon', t, loc)
    sun = get_sun(t)
    sep = moon.separation(star, origin_mismatch='ignore').deg
    elong = sun.separation(moon, origin_mismatch='ignore').rad
    illum = (1 - np.cos(elong)) / 2
    return float(sep), float(illum)

print('Frozen rejection criteria on the new spectra')
for h in HDR:
    sep, ill = moon_check(h['date'])
    h['moon_sep'], h['moon_illum'] = sep, ill
    c4 = h['sn60'] < 20
    c5 = (sep < 30) and (ill > 0.5)
    print(f"  {h['date'][:10]}  SN60 {h['sn60']:5.1f} (C4 {'FAIL' if c4 else 'pass'})  drift 0 (C3 pass; no simultaneous reference)"
          f"  Moon {sep:5.1f} deg, illum {ill:4.2f} (C5 {'FAIL' if c5 else 'pass'})  C1/C2 not applicable (no RVBank flag, no SERVAL)")

# ---------------------------------------------------------------- pre-registered test
P, sP = 4.26837, 0.00027
K, sK = 5.52, 0.84
TC, sTC = 2456298.34, 0.12
NOISE = 3.9
t = np.array([h['bjd'] for h in HDR]); v = np.array([1000 * h['rvc'] for h in HDR]); e = np.array([h['dvrms'] for h in HDR])
wts = np.array([-0.5, 1.0, -0.5])
rng = np.random.default_rng(1)
Pd = rng.normal(P, sP, 200000); Kd = rng.normal(K, sK, 200000); Td = rng.normal(TC, sTC, 200000)
model = -Kd[:, None] * np.sin(2 * np.pi * (t[None, :] - Td[:, None]) / Pd[:, None])
Dpred = model @ wts
Dobs = float(wts @ v)
snoise = float(np.sqrt(np.sum(wts**2 * (NOISE**2 + e**2))))
stot = float(np.sqrt(snoise**2 + Dpred.var()))
ph = ((t - TC) / P) % 1
z = (Dobs - Dpred.mean()) / stot
z0 = Dobs / snoise
print('\nPRE-REGISTERED CONTRAST TEST')
print('  phases from conjunction:', np.round(ph, 3), ' (maximum velocity at 0.75)')
print('  velocities (m/s):', np.round(v, 2), ' DVRMS:', np.round(e, 2))
print(f'  predicted D = {Dpred.mean():+.2f} +- {Dpred.std():.2f} m/s   noise on D = {snoise:.2f} m/s')
print(f'  observed  D = {Dobs:+.2f} m/s')
print(f'  z  = {z:+.2f}   (|z| < 2: consistent with the planet)')
print(f'  z0 = {z0:+.2f}   (against D = 0)')
# power of the test as registered: probability that z0 < -2 if the planet is real
sim = Dpred + rng.normal(0, snoise, Dpred.size)
print(f'  power: P(z0 < -2 | planet) = {np.mean(sim / snoise < -2):.3f};  P(|z| > 2 | planet) = {np.mean(np.abs(sim - Dpred.mean()) / stot > 2):.3f}')
# power anticipated before the pipeline errors were known: v13 numbers (D_pred -5.2 +- 2.7, noise 4.8 m/s),
# and header-BJD prediction with the 3.9 m/s scatter alone
pw_v13 = np.mean((rng.normal(-5.2, 2.7, 200000) + rng.normal(0, 4.8, 200000)) / 4.8 < -2)
sn39 = float(np.sqrt(np.sum(wts**2 * NOISE**2)))
pw_39 = np.mean((Dpred + rng.normal(0, sn39, Dpred.size)) / sn39 < -2)
print(f'  anticipated power: {pw_v13:.3f} (v13 numbers), {pw_39:.3f} (header BJDs, 3.9 m/s only)')
# same with photon-noise errors (CCF NOISE) instead of DVRMS, as a sensitivity check (NOT pre-registered)
e_ph = np.array([1000 * h['noise'] for h in HDR])
sn_ph = float(np.sqrt(np.sum(wts**2 * (NOISE**2 + e_ph**2))))
print(f'  with photon-noise errors: noise on D = {sn_ph:.2f};  z = {(Dobs - Dpred.mean()) / np.sqrt(sn_ph**2 + Dpred.var()):+.2f};  z0 = {Dobs / sn_ph:+.2f}')

out = dict(spectra=HDR, phases=ph.tolist(), D_obs=Dobs, D_pred=float(Dpred.mean()), D_pred_sd=float(Dpred.std()),
           noise_D=snoise, z=float(z), z0=float(z0), noise_D_photon=sn_ph, power_registered=float(np.mean(sim / snoise < -2)),
           power_anticipated_v13=float(pw_v13), power_anticipated_39=float(pw_39),
           z_photon=float((Dobs - Dpred.mean()) / np.sqrt(sn_ph**2 + Dpred.var())), z0_photon=float(Dobs / sn_ph))
json.dump(out, open('heldout_v15.json', 'w'), indent=1)

# ============================================================================ SECONDARY
# NOT pre-registered. Found on 2026-09-29: the ESO HARPS radial-velocity catalogue
# (HARPS_RVCAT_V1, ESO Phase 3) reproduces the RVBank DRS velocities (column RVdrs) of all
# 108 HD 297396 spectra to 0.0004 m/s (see ../heldout/harps_rvcat_v1_hd297396.csv and
# rvcat_compare below). The three new velocities come from the same DRS (HARPS_3.8), the
# same K5 mask and the same STAR,SKY mode as the two 2021 spectra already in RVBank. They
# are therefore on the scale of the post-upgrade RVBank DRS velocities, up to (i) the
# nightly zero point, which RVBank corrects and these nights lack (rms 0.98 m/s), and
# (ii) any instrumental change after 2021, for which we allow an extra 2 m/s.
import pandas as pd
d = pd.read_csv('data/HD297396_rvbank_full.csv')
cat = pd.read_csv('../heldout/harps_rvcat_v1_hd297396.csv')
d['b'] = d.BJD - 2450000
m = pd.merge_asof(cat.sort_values('bjd_m2450000'), d.sort_values('b'), left_on='bjd_m2450000', right_on='b',
                  direction='nearest', tolerance=0.01).dropna(subset=['b'])
dd = (m.rvc_m16000_ms + 16000) - m.RVdrs
print(f'\nSECONDARY (not pre-registered)\n  ESO RVCAT vs RVBank RVdrs: {len(m)} spectra, max |diff| = {np.abs(dd).max():.4f} m/s')
post = d[(d.BJD > 2457170)].copy()
post['night'] = np.floor(post.BJD - 0.5)
g = post.groupby('night')
tb = g.apply(lambda x: np.sum(x.BJD / x.e_RVdrs**2) / np.sum(1 / x.e_RVdrs**2)).values
vb = g.apply(lambda x: np.sum(x.RVdrs / x.e_RVdrs**2) / np.sum(1 / x.e_RVdrs**2)).values
eb = g.apply(lambda x: 1 / np.sqrt(np.sum(1 / x.e_RVdrs**2))).values
JIT = 3.9; NZP = float(d[d.BJD > 2457170].NZPdrs.std()); ZSH = 2.0   # nightly zero-point rms (post-upgrade RVBank nights); shared instrumental allowance
def kep(tt, Pp, Kk, Tt): return -Kk * np.sin(2 * np.pi * (tt - Tt) / Pp)
res = {}
for name, planet in [('planet', True), ('null', False)]:
    lp = []
    for i in range(20000 if planet else 1):
        Pp, Kk, Tt = (Pd[i], Kd[i], Td[i]) if planet else (P, 0.0, TC)
        w = 1 / (eb**2 + JIT**2)
        off = np.sum(w * (vb - kep(tb, Pp, Kk, Tt))) / w.sum(); soff = 1 / np.sqrt(w.sum())
        mu = off + kep(t, Pp, Kk, Tt)
        C = np.diag(e**2 + JIT**2 + NZP**2) + (soff**2 + ZSH**2)  # nightly zero point on the diagonal; offset error and instrumental term shared
        r = v - mu
        lp.append(-0.5 * r @ np.linalg.solve(C, r) - 0.5 * np.linalg.slogdet(2 * np.pi * C)[1])
    lp = np.array(lp)
    res[name] = float(np.log(np.mean(np.exp(lp - lp.max()))) + lp.max())
    if planet:
        print(f'  planet: predicted new velocities (mean) = {np.round(off + kep(t, P, K, TC), 2)}; offset {off:.2f} +- {soff:.2f} m/s')
    else:
        print(f'  null:   offset {off:.2f} +- {soff:.2f} m/s; observed - offset = {np.round(v - off, 2)}')
lnB = res['planet'] - res['null']
print(f'  out-of-sample log Bayes factor ln p(new | planet) - ln p(new | no planet) = {lnB:+.2f}  (odds {np.exp(lnB):.2f}:1)')
out.update(rvcat_max_diff=float(np.abs(dd).max()), n_rvcat=int(len(m)), lnB_new=lnB, nzp_rms_post=NZP)
json.dump(out, open('heldout_v15.json', 'w'), indent=1)
