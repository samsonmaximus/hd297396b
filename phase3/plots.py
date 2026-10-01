"""Phase 3 figures."""
import json, os, pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
import p3data, p3model, p3run, report

FIG = "out/figs"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 140,
                     "axes.grid": True, "grid.alpha": .25})

def fig_ladder():
    L = json.load(open("out/ladder.json"))
    models = ["1p1c", "1p", "2p1c2c", "1p1c_alias", "1p1c-toi"]
    lab = {"1p1c": "1p1c", "1p": "1p (ecc)", "2p1c2c": "2p1c2c",
           "1p1c_alias": "1p1c @ 1.3013 d", "1p1c-toi": "1p1c + TOI-6263.01"}
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    kerns = [("m32", "Matern-3/2 GP", "C0"), ("qp", "quasi-periodic GP", "C1"),
             ("white", "white noise", "C7")]
    w = 0.26
    for j, (k, nm, c) in enumerate(kerns):
        b = f"{k}_0p_serval"
        if b not in L: continue
        z0 = L[b]["logz"]
        xs, ys, es = [], [], []
        for i, mo in enumerate(models):
            t = f"{k}_{mo}_serval"
            if t not in L: continue
            xs.append(i + (j - 1) * w)
            ys.append(L[t]["logz"] - z0 + report.fix_pcorr(t, L[t]))
            es.append(np.hypot(L[t]["logzerr"], L[b]["logzerr"]))
        ax.bar(xs, ys, w, yerr=es, color=c, label=nm, capsize=2)
    ax.axhline(0, color="k", lw=.8)
    ax.axhline(6, color="r", ls="--", lw=.8)
    ax.text(len(models) - .5, 6.4, r"$\Delta\ln Z=6$", color="r", ha="right", fontsize=8)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([lab[m] for m in models])
    ax.set_ylabel(r"$\Delta\ln Z$ vs 0p (wide-prior corrected)")
    ax.set_title("HD 297396 — model ladder, SERVAL RVs + dLW, joint GP")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{FIG}/ladder.png"); plt.close(fig)

def fig_pscan():
    from pscan import gls_scan, fit_0p
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    for ax, pl in zip(axs, ("drs", "serval")):
        d = p3data.build(pl, clip=4.0); S, hp = fit_0p(d)
        T = d["t"].max() - d["t"].min()
        f = np.arange(1 / 10.0, 1 / 1.05, 1.0 / (20 * T)); P = 1 / f
        dl = gls_scan(d["t"], d["rv"], d["erv"], d["epoch"], S, P)
        # plot the upper envelope: with 1e5 grid points a raw line loses the
        # narrow peaks to pixel undersampling
        nb = 3000
        idx = np.linspace(0, len(P), nb + 1).astype(int)
        Pe = np.array([P[a:b].mean() for a, b in zip(idx[:-1], idx[1:]) if b > a])
        De = np.array([dl[a:b].max() for a, b in zip(idx[:-1], idx[1:]) if b > a])
        ax.plot(Pe, De, lw=.7, color="C0")
        for Pk, col, nm in ((4.26836, "g", "4.26836 d"), (1.301301, "r", "1.30130 d")):
            j = np.argmin(np.abs(P - Pk))
            w = np.abs(P - Pk) < 0.01
            h = dl[w].max()
            ax.axvline(Pk, color=col, lw=1, ls="-" if col == "g" else "--", alpha=.7)
            ax.plot([Pk], [h], "v", color=col, ms=5)
            ax.annotate(f"{nm}\n$\\Delta\\ln L$={h:.1f}", (Pk, h),
                        textcoords="offset points", xytext=(6 if col == "g" else -6, -4),
                        ha="left" if col == "g" else "right", fontsize=7, color=col)
        ax.set_ylim(0, dl.max() * 1.25)
        ax.set_xscale("log"); ax.set_ylabel(r"$\Delta\ln L$")
        ax.set_title(f"{pl.upper()}  (green 4.26836 d, red 1.30130 d = sidereal-day alias)",
                     fontsize=8)
    axs[1].set_xlabel("period [d]")
    fig.tight_layout(); fig.savefig(f"{FIG}/period_scan.png"); plt.close(fig)

def fig_phase(tag="m32_1p1c_serval", pipeline="serval"):
    r = report.load(tag)
    if r is None: return
    d = p3data.build(pipeline, clip=4.0)
    m = p3model.Joint(d, n_planets=1, gp="m32", p1_prior=p3run.P1_NARROW)
    th = np.median(r["samples"], axis=0)
    x = m.ix
    # GP conditional mean on the RV series, planet + offsets removed
    resid = d["rv"] - m._rv_model(th)
    Kb = m._kbase(th) * th[x["A_rv"]] ** 2
    S = Kb + np.diag(m._diag(th, "rv"))
    gpm = Kb @ cho_solve(cho_factor(S, lower=True), resid)
    off = np.zeros(m.n)
    for k, msk in enumerate(m.Erv): off += th[x[f"g_rv{k}"]] * msk
    clean = d["rv"] - gpm - off
    P, K, ph = th[x["P1"]], th[x["K1"]], th[x["ph1"]]
    phase = np.mod((d["t"] - m.tref) / P - ph, 1.0)
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0))
    axs[0].errorbar(d["t"] - 2450000, d["rv"], np.sqrt(m._diag(th, "rv")), fmt=".",
                    ms=3, lw=.6, color="0.6", label="RV")
    o = np.argsort(d["t"])
    axs[0].plot(d["t"][o] - 2450000, (gpm + off)[o], "-", color="C3", lw=1, label="GP + offsets")
    axs[0].set_xlabel("BJD - 2450000"); axs[0].set_ylabel("RV [m/s]"); axs[0].legend(fontsize=7)
    pg = np.linspace(0, 1, 300)
    axs[1].errorbar(phase, clean, np.sqrt(m._diag(th, "rv")), fmt=".", ms=3, lw=.6,
                    color="0.75", zorder=1)
    nb = 10
    e2 = m._diag(th, "rv")
    bp, bm, be = [], [], []
    for i in range(nb):
        sel = (phase >= i / nb) & (phase < (i + 1) / nb)
        if sel.sum() < 2: continue
        w = 1 / e2[sel]
        bp.append((i + .5) / nb); bm.append(np.sum(clean[sel] * w) / w.sum())
        be.append(1 / np.sqrt(w.sum()))
    axs[1].errorbar(bp, bm, be, fmt="o", ms=4, color="C3", zorder=3, lw=1.2,
                    label="binned")
    axs[1].plot(pg, K * np.cos(2 * np.pi * pg), "-", color="C0", lw=1.6, zorder=2)
    axs[1].legend(fontsize=7, loc="lower right")
    axs[1].set_xlabel(f"phase (P = {P:.5f} d)"); axs[1].set_ylabel("RV - GP [m/s]")
    axs[1].set_title(f"K = {K:.2f} m/s", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/phasefold.png"); plt.close(fig)

def fig_stability():
    f = "out/stability_ns.json"
    if not os.path.exists(f): return
    R = json.load(open(f))
    key = lambda r: (r["pipeline"], ["white", "m32", "qp"].index(r["kernel"]),
                     r["ecc"], r["noise"])
    R = sorted(R, key=key)
    lab = [f"{r['pipeline'][:3]} {r['kernel']} {'ecc' if r['ecc'] else 'circ'} "
           f"{'clip' if r['noise']=='clip' else 'stud-t'}" for r in R]
    y = np.arange(len(R))
    med = np.array([r["K"][1] for r in R])
    lo = med - np.array([r["K"][0] for r in R]); hi = np.array([r["K"][2] for r in R]) - med
    fig, ax = plt.subplots(figsize=(6.0, 0.24 * len(R) + 1.2))
    ax.errorbar(med, y, xerr=[lo, hi], fmt="o", ms=3, lw=1, color="C0")
    ax.axvline(np.median(med), color="C3", ls="--", lw=.8)
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("K [m/s]")
    ax.set_title("K stability across analysis choices", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/k_stability.png"); plt.close(fig)

if __name__ == "__main__":
    for fn in (fig_ladder, fig_pscan, fig_phase, fig_stability):
        try: fn(); print("ok", fn.__name__)
        except Exception as e: print("FAIL", fn.__name__, type(e).__name__, e)
