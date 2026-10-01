"""Galactic kinematics, population membership and an activity-rotation estimate
for HD 297396, from quantities already in the paper.

Nothing here needs new data: the astrometry is Gaia DR3 (Table 3), the systemic
velocity is the HARPS DRS absolute value, and the activity index is RVBank's.

Constants and relations, each with its source:
  solar peculiar motion   Schoenrich, Binney & Dehnen (2010), MNRAS 403, 1829
                          (U,V,W)_sun = (11.10 +0.69/-0.75, 12.24 +/- 0.47,
                          7.25 +0.37/-0.36) km/s, plus systematics ~(1,2,0.5)
  sign convention         Dehnen & Binney (1998): U toward the Galactic centre
  population ellipsoids   Bensby, Feltzing & Oey (2014), A&A 562, A71, Table A.1
  convective turnover     Noyes et al. (1984), ApJ 279, 763
"""
import json
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactic, ICRS

RA, DEC = "09:16:47.93", "-49:18:03.0"          # Gaia DR3, J2016
PMRA, EPMRA = -214.415, 0.014                    # mas/yr, Gaia DR3
PMDEC, EPMDEC = 116.833, 0.013
PLX, EPLX = 21.6610, 0.0129                      # mas, Gaia DR3 + Lindegren ZP
VSYS, EVSYS = 16.385, 0.100                      # km/s, HARPS DRS absolute
BV = 1.08                                        # TIC / SIMBAD
LOGRHK = -4.64                                   # RVBank

SUN = np.array([11.10, 12.24, 7.25])             # SBD10
SUN_ERR = np.array([np.hypot(0.72, 1.0), np.hypot(0.47, 2.0), np.hypot(0.37, 0.5)])

# Bensby+2014 Table A.1: sigma_U, sigma_V, sigma_W, U_asym, V_asym, X
POP = {
    "thin":  dict(s=(35., 20., 16.),  ua=0.,   va=-15.,  X=0.85),
    "thick": dict(s=(67., 38., 35.),  ua=0.,   va=-46.,  X=0.09),
    "halo":  dict(s=(160., 90., 90.), ua=0.,   va=-220., X=0.0015),
}


def uvw(pmra, pmdec, plx, rv):
    c = SkyCoord(ra=RA, dec=DEC, unit=(u.hourangle, u.deg), frame=ICRS,
                 distance=(1000.0 / plx) * u.pc,
                 pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                 radial_velocity=rv * u.km / u.s)
    g = c.transform_to(Galactic())
    v = g.velocity.d_xyz.to(u.km / u.s).value   # right-handed, +x to GC
    return np.array([v[0], v[1], v[2]])


def f_pop(U, V, W, p):
    sU, sV, sW = p["s"]
    k = 1.0 / ((2 * np.pi) ** 1.5 * sU * sV * sW)
    return k * np.exp(-((U - p["ua"]) ** 2) / (2 * sU ** 2)
                      - ((V - p["va"]) ** 2) / (2 * sV ** 2)
                      - (W ** 2) / (2 * sW ** 2))


def tau_c(bv):
    """Noyes et al. (1984) convective turnover time, days."""
    x = 1.0 - bv
    if x > 0:
        return 10 ** (1.362 - 0.166 * x + 0.025 * x ** 2 - 5.323 * x ** 3)
    return 10 ** (1.362 - 0.14 * x)


def rossby(logrhk):
    """Noyes et al. (1984) Rossby number from log R'HK."""
    y = 5.0 + logrhk
    return 10 ** (0.324 - 0.400 * y - 0.283 * y ** 2 - 1.325 * y ** 3)


rng = np.random.default_rng(20260909)
NS = 200000
pm1 = rng.normal(PMRA, EPMRA, NS)
pm2 = rng.normal(PMDEC, EPMDEC, NS)
px = rng.normal(PLX, EPLX, NS)
rv = rng.normal(VSYS, EVSYS, NS)
# the astrometric errors are negligible here, so sample the solar motion too
sun = rng.normal(SUN, SUN_ERR, (NS, 3))

base = uvw(PMRA, PMDEC, PLX, VSYS)
# linear propagation: recompute at +-1 sigma of each input (errors are tiny)
J = []
for i, (val, err, args) in enumerate((
        (PMRA, EPMRA, 0), (PMDEC, EPMDEC, 1), (PLX, EPLX, 2), (VSYS, EVSYS, 3))):
    a = [PMRA, PMDEC, PLX, VSYS]
    a[args] = val + err
    J.append(uvw(*a) - base)
J = np.array(J)                                  # 4 x 3
helio_draws = base + (rng.standard_normal((NS, 4)) @ J)
lsr = helio_draws + sun

U, V, W = lsr[:, 0], lsr[:, 1], lsr[:, 2]
fs = {k: f_pop(U, V, W, p) for k, p in POP.items()}
TD_D = (POP["thick"]["X"] / POP["thin"]["X"]) * fs["thick"] / fs["thin"]
TD_H = (POP["thick"]["X"] / POP["halo"]["X"]) * fs["thick"] / fs["halo"]
vtot = np.sqrt(U ** 2 + V ** 2 + W ** 2)
toomre = np.sqrt(U ** 2 + W ** 2)

tc = tau_c(BV)
Ro = rossby(LOGRHK)
Prot = Ro * tc
# 0.1 dex scatter in Ro is the usual quoted spread of the activity-rotation relation
Prot_lo, Prot_hi = Prot * 10 ** -0.1, Prot * 10 ** 0.1

def q(a, dp=2):
    lo, m, hi = np.percentile(a, [16, 50, 84])
    return f"{m:.{dp}f} +{hi-m:.{dp}f}/-{m-lo:.{dp}f}"

res = dict(
    U_helio=float(base[0]), V_helio=float(base[1]), W_helio=float(base[2]),
    U_lsr=[float(x) for x in np.percentile(U, [16, 50, 84])],
    V_lsr=[float(x) for x in np.percentile(V, [16, 50, 84])],
    W_lsr=[float(x) for x in np.percentile(W, [16, 50, 84])],
    v_tot=[float(x) for x in np.percentile(vtot, [16, 50, 84])],
    toomre=[float(x) for x in np.percentile(toomre, [16, 50, 84])],
    TD_D=[float(x) for x in np.percentile(TD_D, [16, 50, 84])],
    TD_H=[float(x) for x in np.percentile(TD_H, [16, 50, 84])],
    p_thinlike=float(np.mean(TD_D < 0.5)), p_thicklike=float(np.mean(TD_D > 2.0)),
    dist_pc=1000.0 / PLX,
    tau_c_d=float(tc), Ro=float(Ro), Prot_d=float(Prot),
    Prot_range=[float(Prot_lo), float(Prot_hi)],
    Prot_over_P0=float(Prot / 4.26838),
    BV=BV, logRHK=LOGRHK,
)
json.dump(res, open("out/kinematics.json", "w"), indent=1)

print(f"heliocentric  (U,V,W) = ({base[0]:+.2f}, {base[1]:+.2f}, {base[2]:+.2f}) km/s")
print(f"LSR           U = {q(U)}   V = {q(V)}   W = {q(W)}  km/s")
print(f"|v_LSR|       {q(vtot)} km/s ;  sqrt(U^2+W^2) = {q(toomre)} km/s")
print(f"TD/D          {q(TD_D,4)}    -> thin-disk criterion TD/D < 0.5 : "
      f"P = {res['p_thinlike']:.4f}")
print(f"TD/H          {q(TD_H,1)}")
print(f"tau_c({BV})   {tc:.2f} d ;  Ro = {Ro:.3f} ;  P_rot = {Prot:.1f} d "
      f"({Prot_lo:.1f}-{Prot_hi:.1f} d at 0.1 dex)")
print(f"P_rot / P0    {Prot/4.26838:.2f}   (6P0 = 25.61 d, 7P0 = 29.88 d, 8P0 = 34.15 d)")
