"""Generate the run-derived LaTeX tables for the paper."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "phase6", "overleaf")
RUNS = json.load(open(f"{HERE}/out/phase3_final.json"))

W = {}
for sub in ("phase1/runs", "phase4/runs"):
    p = f"{ROOT}/{sub}"
    if os.path.isdir(p):
        for f in os.listdir(p):
            W[f[:-5]] = json.load(open(f"{p}/{f}"))


def wz(k):
    return W[k]["lnZ"] if k in W else None


def wze(k):
    return W[k]["lnZerr"] if k in W else None


def fmt(v, e=None, dp=2, plus=False):
    if v is None:
        return "\\ldots"
    s = f"{v:+.{dp}f}" if plus else f"{v:.{dp}f}"
    return f"${s}\\pm{e:.{dp}f}$" if e is not None else f"${s}$"


def wpair(a, b):
    if a not in W or b not in W:
        return "\\ldots"
    return fmt(W[a]["lnZ"] - W[b]["lnZ"], float(np.hypot(W[a]["lnZerr"], W[b]["lnZerr"])), plus=True)


def wK(k):
    if k not in W or "K_p1" not in W[k]["params"]:
        return "\\ldots"
    p = W[k]["params"]["K_p1"]
    return f"${p['med']:.2f}^{{+{p['hi']:.2f}}}_{{-{p['lo']:.2f}}}$"


# --------------------------------------------------------- Table: white -----
cols = ["serval_in", "serval_out", "drs_in", "drs_out"]
rows = [("0p (offsets + jitter)", "0p"),
        ("1p1c (circular)", "1p1c"),
        ("1p (eccentric)", "1p_ecc"),
        ("2p1c2c", "2p1c2c")]
L = [r"""\begin{table*}
\caption{Bayesian evidences under the white-noise likelihood, from
\texttt{juliet}. Four models in four data sets: two independent reductions of
the same spectra, each with and without the single discrepant epoch of
Sect.~\ref{sec:plus51} that passes all five pre-registered criteria.}
\label{tab:evidence}
\centering
\begin{tabular}{lcccc}
\hline\hline
Model & \texttt{SERVAL}, epoch in & \texttt{SERVAL}, epoch out & DRS, epoch in & DRS, epoch out \\
\hline"""]
for lab, m in rows:
    cells = [fmt(wz(f"{c}_{m}"), wze(f"{c}_{m}")) for c in cols]
    L.append(f"{lab:<26} & " + " & ".join(cells) + r" \\")
L.append(r"\hline")
L.append(r"$\dlnz$ (1p1c $-$ 0p)     & " + " & ".join(wpair(f"{c}_1p1c", f"{c}_0p") for c in cols) + r" \\")
L.append(r"$\dlnz$ (2p $-$ 1p1c)     & " + " & ".join(wpair(f"{c}_2p1c2c", f"{c}_1p1c") for c in cols) + r" \\")
L.append(r"$K_1$ (m\,s$^{-1}$)       & " + " & ".join(wK(f"{c}_1p1c") for c in cols) + r" \\")
L.append(r"""\hline
\multicolumn{5}{l}{Matched-prior alias pair (\texttt{SERVAL}, epoch in, identical 0.06\,d prior widths):} \\
$P\sim\mathcal{U}(4.238,4.298)$ &  \multicolumn{2}{l}{$\lnz=-366.11\pm0.39$} & \multicolumn{2}{l}{$K=5.59^{+0.91}_{-0.89}\ms$} \\
$P\sim\mathcal{U}(1.276,1.336)$ &  \multicolumn{2}{l}{$\lnz=-376.10\pm0.37$} & \multicolumn{2}{l}{$\dlnz=+9.99\pm0.54$} \\
\hline
\end{tabular}
\tablefoot{Independent offset and jitter per instrument label; no
correlated-noise term. These evidences are superseded by
Table~\ref{tab:evidence3} but are given because they are what a standard
analysis of these data would report. The matched-prior run at 4.268\,d and the
1p1c run differ only in the width of the period prior (0.06 versus 0.14\,d), so
the analytic prior-volume ratio predicts $\lnz = -367.31$ against the
$-366.11$ obtained; the 1.2-nat discrepancy is about twice the combined
run-to-run scatter and is the reason the alias test is repeated under the
correlated-noise model in Sect.~\ref{sec:alias}.}
\end{table*}""")
open(f"{OUT}/tab_evidence.tex", "w").write("\n".join(L) + "\n")

# --------------------------------------- Table: correlated-noise ladder -----
def find(pipeline, plus51, kernel, like, model, coupling="shared_hypers", seeing=None):
    for t, r in RUNS.items():
        if (r["pipeline"] == pipeline and r["plus51"] == plus51 and r["kernel"] == kernel
                and r["like"] == like and r["model"] == model
                and r["coupling"] == coupling and r["seeing"] == seeing
                and not t.startswith("ver_")):
            return t
    return None


def cell(pipeline, plus51, kernel, like, model, **kw):
    t = find(pipeline, plus51, kernel, like, model, **kw)
    if t is None:
        return "\\ldots"
    r = RUNS[t]
    return f"${r['logz']:.2f}$"


def dcell(pipeline, plus51, kernel, like, num, den, wide=False, **kw):
    a = find(pipeline, plus51, kernel, like, num, **kw)
    b = find(pipeline, plus51, kernel, like, den, **kw)
    if a is None or b is None:
        return "\\ldots"
    ra, rb = RUNS[a], RUNS[b]
    v = (ra["logz"] + ra["pcorr"]) - (rb["logz"] + rb["pcorr"]) if wide else ra["logz"] - rb["logz"]
    e = float(np.hypot(ra["logzerr"], rb["logzerr"]))
    return f"${v:+.2f}\\pm{e:.2f}$"


def kcell(pipeline, plus51, kernel, like, model, **kw):
    t = find(pipeline, plus51, kernel, like, model, **kw)
    if t is None or "K1" not in RUNS[t]["summary"]:
        return "\\ldots"
    lo, m, hi = RUNS[t]["summary"]["K1"]
    return f"${m:.2f}\\pm{0.5*(hi-lo):.2f}$"


SETS = [("\\texttt{SERVAL}, in", "serval", True), ("\\texttt{SERVAL}, out", "serval", False),
        ("DRS, in", "drs", True), ("DRS, out", "drs", False)]
NOISE = [("white (this code)", "white", "gauss", {}),
         ("Mat\\'ern GP", "m32", "gauss", {}),
         ("Mat\\'ern GP + Student-$t$", "m32", "studentt", {}),
         ("quasi-periodic GP", "qp", "gauss", {}),
         ("\\textbf{GP + seeing (power)}", "m32", "gauss", dict(seeing="power")),
         ("GP + seeing (hinge)", "m32", "gauss", dict(seeing="hinge")),
         ("GP + seeing (step $1\\farcs5$)", "m32", "gauss", dict(seeing="thresh@1.50")),
         ("GP + seeing + Student-$t$", "m32", "studentt", dict(seeing="power")),
         ("GP, shared latent", "m32", "gauss", dict(coupling="shared_latent")),
         ("GP + seeing, shared latent", "m32", "gauss",
          dict(coupling="shared_latent", seeing="power"))]
L = [r"""\begin{table*}
\caption{The correlated-noise ladder on the frozen data set. All values are
$\dlnz(\mathrm{1p1c}-\mathrm{0p})$ computed through the same code, so
differences between rows isolate the noise model and differences between columns
isolate the data. Uncertainties are propagated internal estimates and should be
read as $\pm0.3$--$0.4$ (Appendix~\ref{app:sampler}).}
\label{tab:evidence3}
\centering
\begin{tabular}{lcccc}
\hline\hline
Noise model & \texttt{SERVAL}, in & \texttt{SERVAL}, out & DRS, in & DRS, out \\
\hline"""]
for lab, kern, like, kw in NOISE:
    L.append(f"{lab:<32} & " + " & ".join(
        dcell(p, i, kern, like, "1p1c", "0p", **kw) for _, p, i in SETS) + r" \\")
L.append(r"\hline")
L.append(r"\multicolumn{5}{l}{\emph{Semi-amplitude $K$ (m\,s$^{-1}$), same cells:}}\\")
for lab, kern, like, kw in (NOISE[0], NOISE[1], NOISE[2], NOISE[4]):
    L.append(f"{lab:<32} & " + " & ".join(
        kcell(p, i, kern, like, "1p1c", **kw) for _, p, i in SETS) + r" \\")
L.append(r"\hline")
L.append(r"\multicolumn{5}{l}{\emph{$\dlnz(\mathrm{2p1c2c}-\mathrm{1p1c})$:}}\\")
for lab, kern, like, kw in (NOISE[0], NOISE[1], NOISE[2], NOISE[4]):
    L.append(f"{lab:<32} & " + " & ".join(
        dcell(p, i, kern, like, "2p1c2c", "1p1c", **kw) for _, p, i in SETS) + r" \\")
L.append(r"""\hline
\end{tabular}
\tablefoot{Evidences are computed with narrow period priors,
$\mathcal{J}(4.20,4.34)$\,d and $\mathcal{J}(150,260)$\,d, and the entries above
are the sampled differences. Correcting to the wide search priors
$\mathcal{J}(1.05,10)$\,d and $\mathcal{J}(10,1000)$\,d subtracts 4.23 from each
$\dlnz(\mathrm{1p1c}-\mathrm{0p})$ and a further 2.13 from each
$\dlnz(\mathrm{2p1c2c}-\mathrm{1p1c})$; the ordering of rows and columns is
unaffected. Uncertainties are \texttt{dynesty}'s propagated internal estimates.
For the rows without a seeing term these agree with the empirical run-to-run
scatter to within a factor of two; for the rows \emph{with} one they understate
it by a factor of about five, and the adopted cell's evidence is quoted in the
text as the mean and standard deviation of independent repeats
(Appendix~\ref{app:sampler}).}
\end{table*}""")
open(f"{OUT}/tab_evidence3.tex", "w").write("\n".join(L) + "\n")

# ------------------------------------------------- Table: noise parameters --
LAB = ["\\texttt{HARPS\\_pre\\_072}", "\\texttt{HARPS\\_pre\\_183}",
       "\\texttt{HARPS\\_pre\\_oth}", "\\texttt{HARPS\\_post}"]
NN = [47, 10, 33, 14]
W1 = [7.997, 6.866, 4.380, 3.951]      # phase-1 white-noise jitter, SERVAL in
cols = [("GP, in", find("serval", True, "m32", "gauss", "1p1c")),
        ("GP + $t$, in", find("serval", True, "m32", "studentt", "1p1c")),
        ("GP + seeing, in", find("serval", True, "m32", "gauss", "1p1c", seeing="power")),
        ("GP, out", find("serval", False, "m32", "gauss", "1p1c")),
        ("GP + $t$, out", find("serval", False, "m32", "studentt", "1p1c"))]
L = [r"""\begin{table*}
\caption{Radial-velocity jitter per instrument label (m\,s$^{-1}$) and GP
hyperparameters, across noise models and both data sets. Two HARPS programmes on
the same star differ by a factor of two in unexplained scatter once the Gaussian
process and the error scaling have had their say, and by a factor of six once
the discrepant epoch is removed.}
\label{tab:noise}
\centering
\begin{tabular}{lccccccc}
\hline\hline
Label & $n$ & white & """ + " & ".join(c[0] for c in cols) + r""" \\
\hline"""]
for i in range(4):
    cells = []
    for _, t in cols:
        if t is None:
            cells.append("\\ldots")
        else:
            lo, m, hi = RUNS[t]["summary"][f"s_rv{i}"]
            cells.append(f"${m:.2f}^{{+{hi-m:.2f}}}_{{-{m-lo:.2f}}}$")
    L.append(f"{LAB[i]} & {NN[i]} & ${W1[i]:.2f}$ & " + " & ".join(cells) + r" \\")
L.append(r"\hline")
extra = []
for nm, key in (("$\\beta_{\\rm RV}$", "beta_rv"), ("$\\beta_{\\Delta \\rm LW}$", "beta_dlw"),
                ("$A_{\\rm RV}$ (m\\,s$^{-1}$)", "A_rv"), ("GP $\\ell$ (d)", "gp_l")):
    cells = []
    for _, t in cols:
        if t is None or key not in RUNS[t]["summary"]:
            cells.append("\\ldots")
        else:
            lo, m, hi = RUNS[t]["summary"][key]
            dp = 0 if key == "gp_l" else 2
            cells.append(f"${m:.{dp}f}^{{+{hi-m:.{dp}f}}}_{{-{m-lo:.{dp}f}}}$")
    extra.append(f"{nm} & & \\ldots & " + " & ".join(cells) + r" \\")
L += extra
L.append(r"""\hline
\end{tabular}
\tablefoot{The Student-$t$ jitters are \emph{larger} than the Gaussian ones, not
smaller: the multivariate Student-$t$ is a global scale mixture and cannot
down-weight a single epoch (Sect.~\ref{sec:robust}).}
\end{table*}""")
open(f"{OUT}/tab_noise.tex", "w").write("\n".join(L) + "\n")
print("wrote tab_evidence.tex, tab_evidence3.tex, tab_noise.tex")
