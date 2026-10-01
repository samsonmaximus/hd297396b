"""Assemble the three Phase 3 outputs from the cached nested-sampling runs."""
import json, os, pickle, numpy as np
import p3data, p3run

MSTAR = 0.776       # Msun, Phase 5 (Gaia DR3 + Kervella+22); was 0.76 in Phase 2
MJ_ME = 317.828

def msini_earth(K, P_days, e=0.0):
    """M sin i in Earth masses for K in m/s, P in days, M_p << M_star."""
    Pyr = P_days / 365.25
    mj = K / 28.4329 * MSTAR ** (2 / 3) * Pyr ** (1 / 3) * np.sqrt(1 - e ** 2)
    return mj * MJ_ME

def load(tag):
    f = f"out/ns_{tag}.pkl"
    return pickle.load(open(f, "rb")) if os.path.exists(f) else None

ALIAS = (1.295, 1.308)
def fix_pcorr(tag, r):
    """Recompute the wide-prior correction from the model's real period window
    (the alias model uses a narrower window than the candidate)."""
    c = 0.0
    lw = np.log(p3run.P1_WIDE[1] / p3run.P1_WIDE[0])
    if "0p" not in tag:
        w = ALIAS if "alias" in tag else p3run.P1_NARROW
        c += np.log(np.log(w[1] / w[0]) / lw)
    if "2p" in tag:
        c += np.log(np.log(p3run.P2_NARROW[1] / p3run.P2_NARROW[0])
                    / np.log(p3run.P2_WIDE[1] / p3run.P2_WIDE[0]))
    return c

def q(x, p=(16, 50, 84)):
    return np.percentile(x, p)

# ---------------------------------------------------------------- 1. ladder --
def ladder_table():
    L = json.load(open("out/ladder.json"))
    lines = []
    for pipe in ("serval", "drs"):
        for kern in ("m32", "qp", "white"):
            base = f"{kern}_0p_{pipe}"
            if base not in L: continue
            z0 = L[base]["logz"]
            lines.append(f"\n  [{pipe.upper()} | {kern}]   lnZ(0p) = {z0:.2f} +- {L[base]['logzerr']:.2f}")
            lines.append(f"  {'model':<14}{'ndim':>5}{'lnZ':>11}{'+-':>7}"
                         f"{'dlnZ vs 0p':>12}{'wide-prior':>12}")
            for mo in ("1p1c", "1p", "2p1c2c", "1p1c_alias", "1p1c-toi"):
                tag = f"{kern}_{mo}_{pipe}"
                if tag not in L: continue
                r = L[tag]
                d = r["logz"] - z0
                pc = fix_pcorr(tag, r)
                lines.append(f"  {mo:<14}{r['ndim']:>5}{r['logz']:>11.2f}{r['logzerr']:>7.2f}"
                             f"{d:>12.2f}{d + pc:>12.2f}")
    return "\n".join(lines)

# --------------------------------------------------------------- 2. jitter --
def jitter_table(tag="m32_1p1c_serval"):
    r = load(tag)
    if r is None: return "(reference run missing)"
    d = p3data.build("serval", clip=4.0)
    nper = np.bincount(d["gidx"], minlength=d["n_group"])
    names = p3data.GROUP_NAMES
    out = [f"  reference run: {tag}",
           f"  {'programme':<18}{'nights':>7}{'jitter_RV [m/s]':>26}{'jitter_dLW':>24}"]
    for k, nm in enumerate(names):
        a = q(r["samples"][:, r["names"].index(f"s_rv{k}")])
        b = q(r["samples"][:, r["names"].index(f"s_dlw{k}")])
        out.append(f"  {nm:<18}{nper[k]:>7}"
                   f"{a[1]:>14.2f} +{a[2]-a[1]:.2f}/-{a[1]-a[0]:.2f}"
                   f"{b[1]:>12.2f} +{b[2]-b[1]:.2f}/-{b[1]-b[0]:.2f}")
    for nm in ("beta_rv", "beta_dlw", "gp_l", "A_rv", "A_dlw"):
        if nm in r["names"]:
            a = q(r["samples"][:, r["names"].index(nm)])
            out.append(f"  {nm:<18}{'':>7}{a[1]:>14.2f} +{a[2]-a[1]:.2f}/-{a[1]-a[0]:.2f}")
    return "\n".join(out)

# ------------------------------------------------------------ 3. stability --
def stability_table():
    f = "out/stability_ns.json"
    if not os.path.exists(f): return "(not run yet)"
    R = json.load(open(f))
    out = [f"  {'pipeline':<9}{'noise-kernel':<14}{'orbit':<6}{'outliers':<10}"
           f"{'n':>4}{'K [m/s]':>22}{'Msini [Me]':>12}"]
    key = lambda r: (r["pipeline"], ["white", "m32", "qp"].index(r["kernel"]),
                     r["ecc"], r["noise"])
    for r in sorted(R, key=key):
        k = r["K"]; e = r.get("e", [0, 0, 0])[1] if r["ecc"] else 0.0
        out.append(f"  {r['pipeline']:<9}{r['kernel']:<14}"
                   f"{'ecc' if r['ecc'] else 'circ':<6}"
                   f"{('4sig clip' if r['noise']=='clip' else 'Student-t'):<10}{r['n']:>4}"
                   f"{k[1]:>10.2f} +{k[2]-k[1]:.2f}/-{k[1]-k[0]:.2f}"
                   f"{msini_earth(k[1], r['P'], e):>12.1f}")
    K = np.array([r["K"][1] for r in R])
    out.append(f"\n  spread across all {len(R)} cells: K = {K.min():.2f} - {K.max():.2f} m/s, "
               f"median {np.median(K):.2f}, sd {K.std(ddof=1):.2f}")
    return "\n".join(out)

# ------------------------------------------------------------------- 4. M5 --
def toi_limit():
    out = []
    for kern in ("m32", "qp"):
        r = load(f"{kern}_1p1c-toi_serval")
        if r is None: continue
        K = r["samples"][:, r["names"].index("K_toi")]
        for cl in (68, 95, 99.7):
            lim = np.percentile(K, cl)
            out.append(f"  {kern}: K_toi < {lim:5.2f} m/s ({cl}%)  ->  "
                       f"M sin i < {msini_earth(lim, p3run.TOI[0]):5.2f} Me")
        out.append(f"  {kern}: K_toi median {np.median(K):.2f} m/s "
                   f"(a 0.93 Re rocky planet would give ~0.35 m/s)")
    return "\n".join(out) or "(not run yet)"

def kipping_table():
    f = "out/stability_kipping.json"
    if not os.path.exists(f): return "(not run yet)"
    R = json.load(open(f))
    U = {(r["pipeline"], r["kernel"], r["noise"]): r
         for r in json.load(open("out/stability_ns.json")) if r["ecc"]}
    out = [f"  {'pipeline':<9}{'kernel':<7}{'outliers':<10}"
           f"{'K (Kipping e)':>20}{'e':>8}{'e95':>7}{'K (uniform e)':>16}{'e':>8}"]
    for r in sorted(R, key=lambda r: (r["pipeline"],
                    ["white", "m32", "qp"].index(r["kernel"]), r["noise"])):
        u = U.get((r["pipeline"], r["kernel"], r["noise"]))
        out.append(f"  {r['pipeline']:<9}{r['kernel']:<7}"
                   f"{('4sig clip' if r['noise']=='clip' else 'Student-t'):<10}"
                   f"{r['K'][1]:>11.2f} +{r['K'][2]-r['K'][1]:.2f}/-{r['K'][1]-r['K'][0]:.2f}"
                   f"{r['e'][1]:>8.3f}{r['e95']:>7.3f}"
                   f"{(u['K'][1] if u else float('nan')):>16.2f}{(u['e'][1] if u else float('nan')):>8.3f}")
    return "\n".join(out)


def verification():
    out = []
    f = "out/verify.json"
    if os.path.exists(f):
        V = json.load(open(f))
        dz = np.array([v["dlnZ"] for v in V] + [24.44])
        out.append("  dlnZ(1p1c - 0p), SERVAL + Matern GP, independent nested-sampling runs:")
        for v in V:
            out.append(f"    seed {v['seed']:>3} nlive {v['nlive']:<5} walks {v['walks']:<3}"
                       f"  dlnZ = {v['dlnZ']:.2f}   K = {v['K'][1]:.2f}"
                       f"   gp_l = {v['gp_l'][1]:.0f} d")
        out.append(f"    (ladder run, nlive 750 walks 30)          dlnZ = 24.44")
        out.append(f"    -> mean {dz.mean():.2f}, sd {dz.std(ddof=1):.2f}, "
                   f"range {dz.min():.2f}-{dz.max():.2f}")
    f = "out/ecc_control.json"
    if os.path.exists(f):
        out.append("\n  eccentric-cell reproducibility (drs / Matern / clip):")
        for r in json.load(open(f)):
            out.append(f"    e-prior {r['e_prior']:<8} seed {r['seed']}  lnZ = {r['logz']:8.2f}"
                       f"  lnLmax = {r['loglmax']:8.2f}  K = {r['K'][1]:.2f}  e = {r['e'][1]:.3f}")
    f = "out/checks.json"
    if os.path.exists(f):
        C = json.load(open(f))
        if "floor10" in C:
            out.append(f"\n  GP timescale prior floor: lnZ = {C['floor50']['logz']:.2f} "
                       f"(floor 50 d) vs {C['floor10']['logz']:.2f} (floor 10 d); "
                       f"K = {C['floor50']['K'][1]:.2f} vs {C['floor10']['K'][1]:.2f}")
    return "\n".join(out) or "(not run yet)"


if __name__ == "__main__":
    print("=" * 78)
    print("PHASE 3 RESULTS  --  HD 297396 (TOI-6263), shared-hyperparameter GP")
    print("=" * 78)
    print("\n[1] MODEL LADDER (log Bayesian evidence, dynesty nested sampling)")
    print(ladder_table())
    print("\n\n[2] JITTER AND NOISE POSTERIORS")
    print(jitter_table())
    print("\n\n[3] K-STABILITY TABLE")
    print(stability_table())
    print("\n\n[3b] ECCENTRIC FITS UNDER THE KIPPING (2013) e PRIOR")
    print(kipping_table())
    print("\n\n[5] VERIFICATION")
    print(verification())
    print("\n\n[4] MASS LIMIT ON THE TRANSITING CANDIDATE TOI-6263.01")
    print(toi_limit())
