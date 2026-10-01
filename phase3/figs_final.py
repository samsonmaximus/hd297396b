"""Figures for the paper. Reads only out/phase3_final.json and the frozen data."""
import json, os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import p3data, p3model, p3run

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "phase6", "overleaf")
RUNS = json.load(open(f"{HERE}/out/phase3_final.json"))
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "figure.dpi": 200,
                     "savefig.bbox": "tight", "mathtext.default": "regular"})
LABCOL = ["#1f4e79", "#c0504d", "#4f8a4f", "#7b5aa6"]
LABNAME = ["HARPS_pre_072", "HARPS_pre_183", "HARPS_pre_oth", "HARPS_post"]


def find(**kw):
    for t, r in RUNS.items():
        if all(r.get(k) == v for k, v in kw.items()) and not t.startswith("ver_"):
            return t
    return None


# ------------------------------------------------------------ fig: ladder ---
def fig_ladder():
    sets = [("SERVAL\nepoch in", "serval", True), ("SERVAL\nepoch out", "serval", False),
            ("DRS\nepoch in", "drs", True), ("DRS\nepoch out", "drs", False)]
    noise = [("white", dict(kernel="white", like="gauss", coupling="shared_hypers", seeing=None), "#888888"),
             ("Matern GP", dict(kernel="m32", like="gauss", coupling="shared_hypers", seeing=None), "#1f4e79"),
             ("QP GP", dict(kernel="qp", like="gauss", coupling="shared_hypers", seeing=None), "#4f8a4f"),
             ("GP + Student-t", dict(kernel="m32", like="studentt", coupling="shared_hypers", seeing=None), "#c0504d"),
             ("GP, shared latent", dict(kernel="m32", like="gauss", coupling="shared_latent", seeing=None), "#7b5aa6")]
    see = sorted({r["seeing"] for r in RUNS.values() if r["seeing"]})
    for i, s in enumerate(see):
        noise.append((f"GP + seeing ({s})",
                      dict(kernel="m32", like="gauss", coupling="shared_hypers", seeing=s),
                      plt.cm.autumn(0.15 + 0.25 * i)))
    fig, ax = plt.subplots(figsize=(3.4, 0.42 * len(noise) + 1.3))
    w = 0.8 / len(sets)
    for j, (slab, pipe, p51) in enumerate(sets):
        ys, xs, es = [], [], []
        for i, (nlab, kw, c) in enumerate(noise):
            a = find(pipeline=pipe, plus51=p51, model="1p1c", **kw)
            b = find(pipeline=pipe, plus51=p51, model="0p", **kw)
            if a is None or b is None:
                continue
            ys.append(i + (j - 1.5) * w)
            xs.append(RUNS[a]["logz"] - RUNS[b]["logz"])
            es.append(float(np.hypot(RUNS[a]["logzerr"], RUNS[b]["logzerr"])))
        ax.errorbar(xs, ys, xerr=es, fmt="o", ms=3.2, lw=0.9, capsize=1.6,
                    color=["#1f4e79", "#7fa8d0", "#c0504d", "#e2a09e"][j], label=slab.replace("\n", " "))
    for x, lab in ((3, None), (6, None), (11, None)):
        ax.axvline(x, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0, color="k", lw=0.7, zorder=0)
    ax.set_yticks(range(len(noise)))
    ax.set_yticklabels([n[0] for n in noise])
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta\ln\mathcal{Z}$ (1p1c $-$ 0p), narrow period prior")
    ax.legend(fontsize=6, frameon=False, loc="lower right", ncol=2)
    ax.grid(axis="x", ls=":", lw=0.4, alpha=0.5)
    fig.savefig(f"{FIG}/fig_ladder.png")
    plt.close(fig)
    print("fig_ladder.png")


# --------------------------------------------------------- fig: stability ---
def fig_stability(adopt):
    rows = []
    for t, r in RUNS.items():
        if "K1" not in r["summary"] or r["model"] == "1p1c_alias" or t.startswith("ver_"):
            continue
        lo, m, hi = r["summary"]["K1"]
        lab = (f"{r['pipeline']}/{'in' if r['plus51'] else 'out'}/{r['kernel']}"
               f"/{'t' if r['like']=='studentt' else 'G'}/{r['model']}"
               + ("/latent" if r["coupling"] == "shared_latent" else "")
               + (f"/see:{r['seeing']}" if r["seeing"] else ""))
        rows.append((lab, m, m - lo, hi - m, t == adopt))
    rows.sort(key=lambda z: z[1])
    fig, ax = plt.subplots(figsize=(3.4, 0.20 * len(rows) + 0.9))
    ka = RUNS[adopt]["summary"]["K1"]
    ax.axvspan(ka[0], ka[2], color="#1f4e79", alpha=0.12, zorder=0)
    ax.axvline(ka[1], color="#1f4e79", lw=0.9, zorder=1)
    for i, (lab, m, a, b, is_ad) in enumerate(rows):
        ax.errorbar(m, i, xerr=[[a], [b]], fmt="o", ms=2.6, lw=0.8, capsize=1.2,
                    color="#c0504d" if is_ad else "0.25", zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.2)
    ax.set_xlabel(r"$K$ (m s$^{-1}$)")
    ax.grid(axis="x", ls=":", lw=0.4, alpha=0.5)
    kv = np.array([r[1] for r in rows])
    ax.set_title(f"spread {kv.min():.2f}-{kv.max():.2f}, sd {kv.std(ddof=1):.2f} m/s "
                 f"({len(kv)} cells)", fontsize=6)
    fig.savefig(f"{FIG}/fig_stability.png")
    plt.close(fig)
    print(f"fig_stability.png ({len(rows)} cells)")


# ------------------------------------------------- model reconstruction -----
def rebuild(tag):
    """Return the data dict, a median-parameter vector and the model object."""
    r = RUNS[tag]
    pipe, p51 = r["pipeline"], r["plus51"]
    d = p3data.build_v1(pipe, p51)
    kw = dict(gp=None if r["kernel"] == "white" else r["kernel"], like=r["like"],
              coupling=r["coupling"], e_prior="kipping",
              p1_prior=p3run.P1_NARROW, p2_prior=p3run.P2_NARROW)
    if r["seeing"]:
        mode = r["seeing"].split("@")[0]
        see = dict(mode=mode, s=p3data.seeing_v1(d["t"]))
        if "@" in r["seeing"]:
            see["thr"] = float(r["seeing"].split("@")[1])
        kw["seeing"] = see
    npl = {"0p": 0, "1p1c": 1, "1p": 1, "2p1c2c": 2}[r["model"]]
    m = p3model.Joint(d, n_planets=npl, ecc=(r["model"] == "1p"), **kw)
    th = np.array([r["summary"][n][1] for n in m.pnames])
    return d, m, th


def gp_mean(m, th, d):
    """Conditional mean of the RV GP given the median parameters."""
    r = d["rv"] - m._rv_model(th)
    dg = m._diag(th, "rv")
    if m.gp is None or m.gp_off:
        return np.zeros_like(r), np.zeros_like(r)
    A = th[m.ix["A_rv"]]
    Kb = A ** 2 * m._kbase(th)
    S = Kb + np.diag(dg)
    from scipy.linalg import cho_factor, cho_solve
    c = cho_factor(S, lower=True)
    mu = Kb @ cho_solve(c, r)
    var = np.diag(Kb) - np.einsum("ij,ji->i", Kb, cho_solve(c, Kb))
    return mu, np.sqrt(np.clip(var, 0, None))


# ------------------------------------------------------------- fig: RV ------
def fig_rv(adopt):
    d, m, th = rebuild(adopt)
    mu, sd = gp_mean(m, th, d)
    t, rv = d["t"], d["rv"]
    off = np.zeros_like(rv)
    for k in range(d["n_epoch"]):
        off += th[m.ix[f"g_rv{k}"]] * (d["epoch"] == k)
    kep = m._rv_model(th) - off
    fig = plt.figure(figsize=(7.1, 4.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.38, wspace=0.24)
    ax = fig.add_subplot(gs[0, :])
    ts = np.linspace(t.min() - 60, t.max() + 60, 4000)
    for k in range(4):
        s = d["gidx"] == k
        ax.errorbar(t[s] - 2450000, rv[s] - off[s], yerr=np.sqrt(m._diag(th, "rv"))[s],
                    fmt="o", ms=2.6, lw=0.6, capsize=0, color=LABCOL[k], label=LABNAME[k], alpha=0.9)
    o = np.argsort(t)
    ax.plot(t[o] - 2450000, mu[o], "-", color="0.25", lw=1.0, label="GP conditional mean")
    ax.fill_between(t[o] - 2450000, (mu - sd)[o], (mu + sd)[o], color="0.5", alpha=0.22, lw=0)
    ax.set_xlabel("BJD $-$ 2450000"); ax.set_ylabel(r"RV (m s$^{-1}$)")
    ax.legend(fontsize=5.6, frameon=False, ncol=5, loc="upper center")
    ax.grid(ls=":", lw=0.4, alpha=0.5)

    P = th[m.ix["P1"]]
    resid = rv - off - mu
    if m.n_planets >= 2:
        resid = resid - (m._rv_model(th) - off - kep)
    ph = ((t - m.tref) / P - th[m.ix["ph1"]]) % 1.0
    ax2 = fig.add_subplot(gs[1, 0])
    for k in range(4):
        s = d["gidx"] == k
        ax2.errorbar(ph[s], resid[s], yerr=np.sqrt(m._diag(th, "rv"))[s], fmt="o", ms=2.2,
                     lw=0.5, capsize=0, color=LABCOL[k], alpha=0.5)
    g = np.linspace(0, 1, 400)
    ax2.plot(g, th[m.ix["K1"]] * np.cos(2 * np.pi * g), "-", color="k", lw=1.1)
    nb = 10
    bi = np.clip((ph * nb).astype(int), 0, nb - 1)
    w = 1.0 / m._diag(th, "rv")
    bx = np.array([np.average(ph[bi == j], weights=w[bi == j]) for j in range(nb)])
    by = np.array([np.average(resid[bi == j], weights=w[bi == j]) for j in range(nb)])
    be = np.array([1 / np.sqrt(w[bi == j].sum()) for j in range(nb)])
    ax2.errorbar(bx, by, yerr=be, fmt="s", ms=4, color="#c0504d", mec="k", mew=0.4,
                 lw=1.0, capsize=2, zorder=5)
    ax2.set_xlabel("orbital phase"); ax2.set_ylabel(r"RV $-$ GP (m s$^{-1}$)")
    ax2.set_title(f"P = {P:.5f} d,  K = {th[m.ix['K1']]:.2f} m/s", fontsize=6.5)
    ax2.grid(ls=":", lw=0.4, alpha=0.5)

    ax3 = fig.add_subplot(gs[1, 1])
    res2 = resid - th[m.ix["K1"]] * np.cos(2 * np.pi * (ph))
    for k in range(4):
        s = d["gidx"] == k
        ax3.errorbar(ph[s], res2[s], yerr=np.sqrt(m._diag(th, "rv"))[s], fmt="o", ms=2.2,
                     lw=0.5, capsize=0, color=LABCOL[k], alpha=0.6)
    ax3.axhline(0, color="k", lw=0.7)
    ax3.set_xlabel("orbital phase"); ax3.set_ylabel(r"residual (m s$^{-1}$)")
    ax3.set_title(f"wrms = {np.sqrt(np.average(res2**2, weights=w)):.2f} m/s", fontsize=6.5)
    ax3.grid(ls=":", lw=0.4, alpha=0.5)
    fig.savefig(f"{FIG}/fig_rv.png")
    plt.close(fig)
    print("fig_rv.png")


# -------------------------------------------------------- fig: periodogram --
def gls(t, y, w, freqs):
    """Weighted Delta chi^2 periodogram with a floating mean."""
    yp = y - np.average(y, weights=w)
    out = np.empty(len(freqs))
    for a in range(0, len(freqs), 3000):
        f = freqs[a:a + 3000][:, None]
        ph = 2 * np.pi * f * t[None, :]
        C, S = np.cos(ph), np.sin(ph)
        C = C - (w * C).sum(1)[:, None] / w.sum()
        S = S - (w * S).sum(1)[:, None] / w.sum()
        cc = (w * C * C).sum(1); ss = (w * S * S).sum(1); cs = (w * C * S).sum(1)
        cy = (w * C * yp).sum(1); sy = (w * S * yp).sum(1)
        det = np.where(np.abs(cc * ss - cs * cs) < 1e-12, np.nan, cc * ss - cs * cs)
        A = (ss * cy - cs * sy) / det
        B = (-cs * cy + cc * sy) / det
        out[a:a + 3000] = A * cy + B * sy
    return np.nan_to_num(out)


def fig_periodogram(adopt):
    """Periodogram stack in the whitened frame of the adopted covariance.

    The adopted noise model assigns very different variances to different
    epochs, so an ordinary weighted periodogram of these velocities is dominated
    by that weighting and its shuffle-based null is meaningless (permuting
    velocities between epochs breaks the epoch-variance correspondence). Both
    RV panels are therefore computed with the exact generalised-least-squares
    statistic under the full covariance, and their false-alarm levels come from
    draws of that covariance rather than from shuffles.
    """
    from scipy.linalg import cholesky
    d, m, th = rebuild(adopt)
    t = d["t"]; n = len(t); T = t.max() - t.min()
    fr = np.arange(1 / 1000., 0.8, 1 / (4 * T))
    Sig = np.diag(m._diag(th, "rv"))
    if m.gp is not None and not m.gp_off:
        Sig = Sig + th[m.ix["A_rv"]] ** 2 * m._kbase(th)
    L = cholesky(Sig, lower=True); Li = np.linalg.inv(L)
    X0 = np.array([(d["gidx"] == k).astype(float) for k in range(d["n_group"])]).T
    Q, _ = np.linalg.qr(Li @ X0)

    def proj(V):
        Vw = np.atleast_2d(V) @ Li.T
        return Vw - (Vw @ Q) @ Q.T

    C = np.empty((len(fr), n)); S = np.empty((len(fr), n))
    for a in range(0, len(fr), 3000):
        f = fr[a:a + 3000][:, None]; ph = 2 * np.pi * f * t[None, :]
        C[a:a + 3000] = proj(np.cos(ph)); S[a:a + 3000] = proj(np.sin(ph))
    cc = (C * C).sum(1); ss = (S * S).sum(1); cs = (C * S).sum(1)
    det = cc * ss - cs * cs; det[np.abs(det) < 1e-10] = np.inf

    def wgls(y):
        yp = proj(y)[0]
        cy = C @ yp; sy = S @ yp
        A = (ss * cy - cs * sy) / det
        B = (-cs * cy + cc * sy) / det
        return A * cy + B * sy

    off = np.zeros(n)
    for k in range(d["n_epoch"]):
        off += th[m.ix[f"g_rv{k}"]] * (d["epoch"] == k)
    mu, _ = gp_mean(m, th, d)
    raw = d["rv"] - off
    res = d["rv"] - m._rv_model(th) - mu
    rng = np.random.default_rng(7)
    null = np.array([wgls((L @ rng.standard_normal(n))).max() for _ in range(300)])
    thr = float(np.percentile(null, 99))
    win = np.abs(np.exp(-2j * np.pi * fr[:, None] * t[None, :]).sum(1)) ** 2 / len(t) ** 2
    wd = 1.0 / m._diag(th, "dlw")
    dlw_p = gls(t, d["dlw"], wd, fr)
    nulld = np.array([gls(t, rng.permutation(d["dlw"]), wd, fr).max() for _ in range(200)])

    panels = [("RV", wgls(raw), thr), ("RV residuals", wgls(res), thr),
              ("window", win, None), (r"$\Delta$LW", dlw_p, float(np.percentile(nulld, 99)))]
    fig, axes = plt.subplots(4, 1, figsize=(3.4, 5.2), sharex=True)
    for ax, (lab, p, tl) in zip(axes, panels):
        ax.plot(1 / fr, p, "-", color="#1f4e79", lw=0.7)
        if tl is not None:
            ax.axhline(tl, color="#c0504d", ls="--", lw=0.7)
        for P, c in ((4.26836, "#c0504d"), (1.30131, "#7b5aa6"), (200.886, "#4f8a4f")):
            ax.axvline(P, color=c, lw=0.7, alpha=0.6, zorder=0)
        ax.set_xscale("log"); ax.set_ylabel(lab, fontsize=7)
        ax.grid(ls=":", lw=0.4, alpha=0.5)
    axes[-1].set_xlabel("period (d)")
    axes[0].set_title("4.268 d (red), 1.301 d alias (purple), 200.9 d (green)", fontsize=6)
    fig.tight_layout(h_pad=0.35)
    fig.savefig(f"{FIG}/fig_periodogram.png")
    plt.close(fig)
    print("fig_periodogram.png")



# ------------------------------------------------------- fig: completeness --
def fig_completeness():
    """Completeness map from the GP-whitened injection-recovery grid."""
    z = np.load(f"{HERE}/out/injrec_gp.npz", allow_pickle=True)
    P, K, comp = z["Pgrid"], z["Kgrid"], z["comp"]
    adopt = str(z["adopt"])
    d, m, th = rebuild(adopt)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    pm_ = ax.pcolormesh(P, K, comp, cmap="viridis", vmin=0, vmax=1, shading="gouraud")
    cs = ax.contour(P, K, comp, levels=[0.5, 0.95], colors=["white", "#ff4444"],
                    linewidths=1.0)
    ax.clabel(cs, fmt={0.5: "50%", 0.95: "95%"}, fontsize=6)
    ax.plot(th[m.ix["P1"]], th[m.ix["K1"]], "*", ms=13, color="#ffdd33", mec="k", mew=0.7,
            label="HD 297396 b", zorder=5)
    ax.plot(200.9, 4.2, "o", ms=6, mfc="none", mec="w", mew=1.2, zorder=5,
            label="200.9 d residual")
    if "P2" in m.ix:
        ax.plot(th[m.ix["P2"]], th[m.ix["K2"]], "o", ms=6, mfc="none", mec="w", mew=1.2,
                label="200.9 d residual")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("period (d)"); ax.set_ylabel(r"$K$ (m s$^{-1}$)")
    ax.set_yticks([0.5, 1, 2, 5, 10]); ax.set_yticklabels(["0.5", "1", "2", "5", "10"])
    ax.legend(fontsize=6, frameon=False, loc="lower left", labelcolor="w")
    cb = fig.colorbar(pm_, ax=ax, pad=0.02); cb.set_label("recovery fraction", fontsize=7)
    ax.set_title("GP-whitened detection limits, 1% global FAP", fontsize=7)
    fig.savefig(f"{FIG}/fig_completeness.png")
    plt.close(fig)
    print("fig_completeness.png")


if __name__ == "__main__":
    adopt = os.environ.get("ADOPT", "v1_serval_in_studentt_1p1c")
    which = sys.argv[1:] or ["ladder", "stability", "rv", "periodogram"]
    if "ladder" in which: fig_ladder()
    if "stability" in which: fig_stability(adopt)
    if "rv" in which: fig_rv(adopt)
    if "periodogram" in which: fig_periodogram(adopt)
    if "completeness" in which: fig_completeness()
