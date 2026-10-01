"""Phase 3 headline figure: what the correlated-noise model does to the evidence,
on the Phase-1 frozen rows so it is directly comparable to Phases 1 and 4."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": .25, "axes.axisbelow": True})
L = json.load(open("out/ladder_v1.json"))
def dz(ds, like, a, b):
    ta, tb = f"v1_{ds}_{like}_{a}", f"v1_{ds}_{like}_{b}"
    if ta not in L or tb not in L: return None, None
    return (L[ta]["logz"] - L[tb]["logz"],
            float(np.hypot(L[ta]["logzerr"], L[tb]["logzerr"])))

DS = [("serval_in", "SERVAL\n+51 in (primary)"), ("drs_in", "DRS\n+51 in"),
      ("serval_out", "SERVAL\n+51 out")]
P1 = {"serval_in": 6.47, "drs_in": 6.01, "serval_out": 19.71}          # Phase 1, white
P4 = {"serval_in": 3.79, "drs_in": -0.11, "serval_out": 9.43}          # Phase 4, white

fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.6))
w = 0.26
for ax, (a, b, p1, title) in zip(axs, [
        ("1p1c", "0p", P1, r"planet vs no planet:  $\Delta\ln Z$(1p1c $-$ 0p)"),
        ("2p1c2c", "1p1c", P4, r"second signal:  $\Delta\ln Z$(2p1c2c $-$ 1p1c)")]):
    x = np.arange(len(DS))
    ax.bar(x - w, [p1[d] for d, _ in DS], w, color="0.65",
           label="white noise (Phases 1/4)")
    for j, (like, c, nm) in enumerate([("gauss", "C0", "Matérn-3/2 GP"),
                                       ("studentt", "C3", "GP + Student-t")]):
        ys, es, xs = [], [], []
        for i, (d, _) in enumerate(DS):
            v, e = dz(d, like, a, b)
            if v is None: continue
            xs.append(i + j * w); ys.append(v); es.append(e)
        ax.bar(xs, ys, w, yerr=es, capsize=2, color=c, label=nm)
    ax.axhline(0, color="k", lw=.8)
    ax.axhline(6, color="darkred", ls="--", lw=.9)
    ax.text(len(DS) - .45, 6.5, "decisive (6)", color="darkred", fontsize=7.5, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in DS], fontsize=8)
    ax.set_ylabel(r"$\Delta\ln Z$"); ax.set_title(title, fontsize=9)
axs[0].legend(fontsize=7.5, loc="upper left")
fig.suptitle("HD 297396 — Phase 3 noise models on the Phase-1 frozen data (data-v1)",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("out/figs/phase3_v1_ladder.png"); plt.close(fig)
print("wrote out/figs/phase3_v1_ladder.png")
