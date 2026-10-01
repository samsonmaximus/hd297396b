"""
Phase 3 dataset assembly for HD 297396 (TOI-6263).

Builds the stacked [RV, dLW] vector used by the joint GP likelihood:
nightly-binned, per-pipeline RVs plus the SERVAL differential line width,
with pre/post-fibre epoch labels and programme (jitter-group) labels.
"""
import os
import numpy as np, pandas as pd

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

CSV     = os.path.join(_ROOT, "jitter_clean", "hd297396_data_v1.csv")
FIBRE   = 2457174.5          # HARPS fibre upgrade, 2015-06-01 (BJD)
P0      = 4.26836            # Phase-1/2 candidate period
RVCOL   = {"drs": ("RVdrsnzp", "e_RVdrsnzp"), "serval": ("DRVmlcnzp", "e_DRVmlcnzp")}

# Programmes with enough nights to carry their own jitter; the rest are pooled.
# (n_nights: 072.C-0488=47, 183.C-0972=10, 0100.C-0097=8, 078.C-0044=8, rest <=5)
JGROUPS = ["072.C-0488(E)", "183.C-0972(A)", "0100.C-0097(A)", "078.C-0044(A)"]
GROUP_NAMES = JGROUPS + ["other"]


def _nightly(df, ycol, ecol):
    """Inverse-variance nightly binning. Returns t, y, e, prog(first spectrum)."""
    night = np.floor(df["BJD"].values - 0.5).astype(int)
    t, y, e, prog, nspec = [], [], [], [], []
    for n in np.unique(night):
        m = night == n
        w = 1.0 / df[ecol].values[m] ** 2
        t.append(np.sum(df["BJD"].values[m] * w) / w.sum())
        y.append(np.sum(df[ycol].values[m] * w) / w.sum())
        e.append(1.0 / np.sqrt(w.sum()))
        prog.append(df["ProgID"].values[m][0])
        nspec.append(m.sum())
    return (np.array(t), np.array(y), np.array(e),
            np.array(prog, dtype=object), np.array(nspec))


def _circ_fit(t, y, e, epoch):
    """Phase-1 style circular fit at fixed grid of P, with a per-epoch offset.
    Linear in (A, B, offsets) at fixed P, so the linear solve is exact."""
    Pg = np.linspace(P0 * 0.98, P0 * 1.02, 40000)
    cols = [np.ones_like(t)] + [(epoch == k).astype(float) for k in range(1, epoch.max() + 1)]
    best = (np.inf, None)
    for P in Pg:
        ph = 2 * np.pi * t / P
        X = np.vstack([np.sin(ph), np.cos(ph)] + cols).T
        b, *_ = np.linalg.lstsq(X / e[:, None], y / e, rcond=None)
        c2 = np.sum(((y - X @ b) / e) ** 2)
        if c2 < best[0]:
            best = (c2, (P, b, X))
    P, b, X = best[1]
    return P, np.hypot(b[0], b[1]), y - X @ b


def build(pipeline="drs", clip=4.0, apply_reject=False):
    """Return a dict describing the stacked dataset.

    clip: MAD clip (in sigma) applied to residuals of the Phase-1 circular fit;
          0 disables (use with the Student-t likelihood instead).
    apply_reject: drop the 6 spectra the author's v1 QA flagged (seeing / DPR type).
    """
    df = pd.read_csv(CSV)
    df = df[df["Flag"] == 0]
    if apply_reject:
        df = df[~df["reject"].astype(bool)]
    rvc, rve = RVCOL[pipeline]
    df = df.dropna(subset=["BJD", rvc, rve, "dLW", "e_dLW"]).sort_values("BJD")

    t, rv, erv, prog, nspec = _nightly(df, rvc, rve)
    t2, dlw, edlw, _, _     = _nightly(df, "dLW", "e_dLW")
    assert np.allclose(t, t2), "RV and dLW nightly grids disagree"

    rv = rv - np.median(rv)
    dlw = dlw - np.median(dlw)
    epoch = (t >= FIBRE).astype(int)          # 0 = pre-fibre, 1 = post-fibre

    clipped = np.zeros(len(t), bool)
    if clip:
        _, _, res = _circ_fit(t, rv, erv, epoch)
        mad = 1.4826 * np.median(np.abs(res - np.median(res)))
        clipped = np.abs(res - np.median(res)) >= clip * mad
        keep = ~clipped
        t, rv, erv, dlw, edlw, prog, nspec, epoch = (
            a[keep] for a in (t, rv, erv, dlw, edlw, prog, nspec, epoch))
        rv = rv - np.median(rv); dlw = dlw - np.median(dlw)

    gidx = np.array([GROUP_NAMES.index(p) if p in JGROUPS else len(JGROUPS)
                     for p in prog])
    return dict(t=t, rv=rv, erv=erv, dlw=dlw, edlw=edlw, epoch=epoch,
                gidx=gidx, prog=prog, nspec=nspec, pipeline=pipeline,
                n=len(t), n_epoch=int(epoch.max()) + 1, n_group=len(GROUP_NAMES),
                n_clipped=int(clipped.sum()))


if __name__ == "__main__":
    for pl in ("drs", "serval"):
        for cl in (4.0, 0.0):
            d = build(pl, clip=cl)
            P, K, res = _circ_fit(d["t"], d["rv"], d["erv"], d["epoch"])
            w = 1 / d["erv"] ** 2
            wr = np.sqrt(np.sum(w * (res - np.average(res, weights=w)) ** 2) / w.sum())
            print(f"{pl:>7} clip={cl:<4} n={d['n']:<4} dropped={d['n_clipped']} "
                  f" P={P:.5f}  K={K:.2f} m/s  res wrms={wr:.2f}"
                  f"  groups={np.bincount(d['gidx'], minlength=5)}"
                  f"  epochs={np.bincount(d['epoch'])}")


# ---------------------------------------------------------------- data-v1 ---
# The Phase-1 frozen data set (`data-v1`), with its pre-registered, non-circular
# rejection criteria and its four instrument labels. Phase 3's own builder above
# uses a residual-based MAD clip, which Phase 1 deliberately avoided; these
# loaders exist so the Phase 3 ladder can be quoted on the same rows as Phases
# 1 and 4.
V1DIR = os.path.join(_ROOT, "phase1", "data")
V1_LABELS = ["HARPS_pre_072", "HARPS_pre_183", "HARPS_pre_oth", "HARPS_post"]


def _nightly_dlw():
    """Nightly inverse-variance-binned dLW from the RVBank table, keyed by night."""
    df = pd.read_csv(CSV)
    df = df[df["Flag"] == 0].dropna(subset=["BJD", "dLW", "e_dLW"])
    night = np.floor(df["BJD"].values - 0.5).astype(int)
    out = {}
    for n in np.unique(night):
        m = night == n
        w = 1.0 / df["e_dLW"].values[m] ** 2
        out[n] = (np.sum(df["dLW"].values[m] * w) / w.sum(), 1.0 / np.sqrt(w.sum()))
    return out


def build_v1(pipeline="serval", plus51=True):
    """Frozen Phase-1 data set + the matching nightly dLW series.

    plus51=True is the Phase-1 *primary* set (the +51 m/s night kept, because it
    trips none of the five criteria); plus51=False is the Phase-1 sensitivity set.
    Offsets and jitter both use Phase 1's four instrument labels.
    """
    f = f"{V1DIR}/rv_{pipeline}_{'in' if plus51 else 'out'}.csv"
    d = pd.read_csv(f).sort_values("bjd").reset_index(drop=True)
    dl = _nightly_dlw()
    key = np.floor(d["bjd"].values - 0.5).astype(int)
    miss = [k for k in key if k not in dl]
    if miss:
        raise SystemExit(f"no dLW for nights {miss}")
    dlw = np.array([dl[k][0] for k in key])
    edlw = np.array([dl[k][1] for k in key])
    lab = np.array([V1_LABELS.index(s) for s in d["inst"].values])
    rv = d["rv"].values - np.median(d["rv"].values)
    return dict(t=d["bjd"].values, rv=rv, erv=d["rverr"].values,
                dlw=dlw - np.median(dlw), edlw=edlw,
                epoch=lab, gidx=lab, prog=d["inst"].values,
                nspec=d["nspec"].values, pipeline=pipeline + ("_in" if plus51 else "_out"),
                n=len(d), n_epoch=len(V1_LABELS), n_group=len(V1_LABELS), n_clipped=0)


GROUP_NAMES_V1 = V1_LABELS


# ---------------------------------------------------------------- seeing ----
# DIMM seeing from the ESO archive headers, nightly-averaged and joined onto the
# frozen rows by the same night key used for dLW. Phase 0 established from this
# metadata alone -- not from any residual -- that the +51 m/s night was taken at
# 1.97", second-worst of 111 exposures, and that the residual wrms rises from
# 3.40 m/s below 0.8" to 20.59 m/s above 1.5". 85 of 108 spectra carry a value;
# the rest have -1.0 in the header (the DIMM was down over a run of 2008-2013
# nights) and are flagged NaN here rather than imputed.
def seeing_v1(t):
    """Nightly mean DIMM seeing for the nights in `t`; NaN where unmeasured."""
    df = pd.read_csv(CSV)
    df = df[df["Flag"] == 0]
    night = np.floor(df["BJD"].values - 0.5).astype(int)
    s = df["seeing"].values.astype(float)
    s = np.where(np.isfinite(s) & (s > 0), s, np.nan)
    by = {}
    for n in np.unique(night):
        v = s[night == n]
        v = v[np.isfinite(v)]
        by[n] = float(np.mean(v)) if v.size else np.nan
    key = np.floor(np.asarray(t) - 0.5).astype(int)
    return np.array([by.get(k, np.nan) for k in key])
