# The out-of-sample test of Sect. 8.3, written down before the data are read.
#
# Three HARPS spectra of HD 297396 were taken after the HARPS-RVBank release:
# 2022 March 18, 2024 March 20 and 2024 November 15. They were never used in the
# analysis. The adopted model (Table 4: P = 4.26837 +- 0.00027 d, K = 5.52 +- 0.84 m/s,
# T_conj = 2456298.34 +- 0.12, circular) predicts their velocities up to one unknown
# offset. Because the offset is unknown, the test is a CONTRAST:
#
#     D = RV(2024 March) - mean(RV(2022 March), RV(2024 November))
#
# which is offset-free. The prediction and its uncertainty are computed below from the
# header BJDs, by Monte Carlo over the posterior of (P, K, T_conj). The noise per epoch is
# the post-upgrade residual scatter of the adopted fit, 3.9 m/s (Sect. 8.3), plus each
# spectrum's own formal error.
#
# DECISION RULE (fixed here, before reading the velocities):
#   z = (D_obs - D_pred) / sigma_total.  |z| < 2: consistent with the planet.
#   Also report z0 = D_obs / sigma_noise, the significance against "no planet" (D = 0).
#   The test has about 1.2 sigma of power, so it cannot confirm the planet; a result
#   with |z| > 2 would count against it and must be reported.
#
# USAGE
#   Put the three HARPS reduced products in one folder (the ESO archive "ADP" 1-D
#   spectra or the DRS "CCF" files both carry the radial velocity in their headers) and run
#       python heldout_new.py path/to/folder
#   The script reads BJD, RV and RV error from whichever keyword set the files carry
#   (old DRS 3.5: HIERARCH ESO DRS BJD / DRS CCF RVC / DRS DVRMS;
#    new DRS 3.x: HIERARCH ESO QC BJD / QC CCF RV / QC CCF RV ERROR),
#   refuses to mix pipelines, and prints the result.
import sys, glob, os
import numpy as np
try:
    from astropy.io import fits
except ImportError:
    sys.exit('needs astropy:  pip install astropy')

P, sP = 4.26837, 0.00027
K, sK = 5.52, 0.84
TC, sTC = 2456298.34, 0.12
NOISE = 3.9          # m/s per epoch, post-upgrade residual scatter of the adopted fit

KEYSETS = [
    # (bjd, rv [km/s], rv error [km/s], label)
    ('HIERARCH ESO DRS BJD', 'HIERARCH ESO DRS CCF RVC', 'HIERARCH ESO DRS DVRMS', 'DRS 3.5'),
    ('HIERARCH ESO QC BJD', 'HIERARCH ESO QC CCF RV', 'HIERARCH ESO QC CCF RV ERROR', 'DRS 3.x'),
]

def read(path):
    h = fits.getheader(path, 0)
    if h.get('OBJECT', '').replace(' ', '').upper() not in ('HD297396', 'HIP45533', 'TOI-6263', 'TIC293547742'):
        print(f'  note: OBJECT = {h.get("OBJECT")!r} in {os.path.basename(path)}')
    for kb, kr, ke, lab in KEYSETS:
        if kb in h and kr in h:
            e = h.get(ke, np.nan)
            # DRS 3.5 DVRMS is in m/s; QC CCF RV ERROR is in km/s
            e_ms = e if lab == 'DRS 3.5' else 1000 * e
            return dict(file=os.path.basename(path), bjd=float(h[kb]), rv=1000 * float(h[kr]),
                        erv=float(e_ms), pipe=lab, date=h.get('DATE-OBS', '?'))
    return None

def main(folder):
    rows = [r for r in (read(p) for p in sorted(glob.glob(os.path.join(folder, '*.fits')))) if r]
    if not rows:
        sys.exit('no file with a recognised RV keyword set; list the header keywords containing RV and BJD and adapt KEYSETS')
    for r in rows:
        print(f"{r['file']:45s} {r['date']:25s} BJD {r['bjd']:.5f}  RV {r['rv']:10.2f} +- {r['erv']:.2f} m/s  [{r['pipe']}]")
    if len({r['pipe'] for r in rows}) > 1:
        sys.exit('the spectra come from different pipelines: offsets differ, the contrast is not valid')
    # one velocity per night (weighted mean if several spectra)
    nights = {}
    for r in rows:
        nights.setdefault(int(np.floor(r['bjd'] - 0.5)), []).append(r)
    ep = []
    for n, rr in sorted(nights.items()):
        w = np.array([1 / r['erv']**2 for r in rr])
        ep.append((np.sum(w * [r['bjd'] for r in rr]) / w.sum(), np.sum(w * [r['rv'] for r in rr]) / w.sum(), 1 / np.sqrt(w.sum())))
    ep = np.array(ep)
    if len(ep) != 3:
        sys.exit(f'expected three nights, found {len(ep)}; the contrast is defined for 2022 Mar, 2024 Mar, 2024 Nov')
    t, v, e = ep.T
    # the middle epoch (2024 March) against the mean of the other two
    wts = np.array([-0.5, 1.0, -0.5])
    rng = np.random.default_rng(1)
    Pd = rng.normal(P, sP, 200000); Kd = rng.normal(K, sK, 200000); Td = rng.normal(TC, sTC, 200000)
    model = -Kd[:, None] * np.sin(2 * np.pi * (t[None, :] - Td[:, None]) / Pd[:, None])
    Dpred = model @ wts
    Dobs = wts @ v
    snoise = np.sqrt(np.sum(wts**2 * (NOISE**2 + e**2)))
    stot = np.sqrt(snoise**2 + Dpred.var())
    ph = ((t - TC) / P) % 1
    print('\nphases from conjunction:', np.round(ph, 3), ' (maximum velocity at 0.75)')
    print(f'predicted D = {Dpred.mean():+.2f} +- {Dpred.std():.2f} m/s (model)   noise on D = {snoise:.2f} m/s')
    print(f'observed  D = {Dobs:+.2f} m/s')
    print(f'z (observed - predicted) = {(Dobs - Dpred.mean()) / stot:+.2f}   [|z| < 2: consistent with the planet]')
    print(f'z0 (observed vs no planet, D = 0) = {Dobs / snoise:+.2f}')

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
