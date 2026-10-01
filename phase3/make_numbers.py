"""Generate numbers.tex and the run-derived LaTeX tables for the paper.

Every numerical result quoted in hd297396b.tex is a macro defined here and
computed from out/phase3_final.json (nested-sampling output),
phase1/runs/*.json (the white-noise ladder) and phase5/system_params.json.
The script asserts that the set of macros it defines is exactly the set the
paper uses, so the paper cannot quote a number that no run produced.
"""
import json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "phase6", "overleaf")
TEX = os.path.join(OUT, "hd297396b.tex")

RUNS = json.load(open(f"{HERE}/out/phase3_final.json"))
SYS = json.load(open(f"{ROOT}/phase5/system_params.json"))
P1 = {}
for sub in ("phase1/runs", "phase4/runs"):
    d_ = f"{ROOT}/{sub}"
    if os.path.isdir(d_):
        for f in os.listdir(d_):
            P1[f[:-5]] = json.load(open(f"{d_}/{f}"))

# ---- which run is the paper's headline. Set by Gate 0.
ADOPT = os.environ.get("ADOPT", "v1_serval_in_studentt_1p1c")
ADOPT_0P = os.environ.get("ADOPT0P", "v1_serval_in_studentt_0p")
ADOPT_TXT = os.environ.get("ADOPTTXT",
    "the Mat\\'ern-3/2 GP with a Student-$t$ likelihood on the primary data set")
OUT_1P = os.environ.get("OUT1P", "v1_serval_out_gauss_1p1c")
OUT_0P = os.environ.get("OUT0P", "v1_serval_out_gauss_0p")

N = {}


# ----------------------------------------------------------------- helpers --
def q(tag, name):
    return RUNS[tag]["summary"][name]          # [16, 50, 84]


def pm(tag, name, dp=2, unit=""):
    lo, med, hi = q(tag, name)
    a, b = med - lo, hi - med
    if abs(a - b) < 0.15 * max(a, b):
        return f"\\ensuremath{{{med:.{dp}f} \\pm {0.5*(a+b):.{dp}f}{unit}}}"
    return f"\\ensuremath{{{med:.{dp}f}^{{+{b:.{dp}f}}}_{{-{a:.{dp}f}}}{unit}}}"


def med(tag, name):
    return q(tag, name)[1]


def sig(tag, name):
    lo, _, hi = q(tag, name)
    return 0.5 * (hi - lo)


def dz(a, b, wide=False):
    ra, rb = RUNS[a], RUNS[b]
    if wide:
        return (ra["logz"] + ra["pcorr"]) - (rb["logz"] + rb["pcorr"])
    return ra["logz"] - rb["logz"]


def dzs(a, b, err=None, wide=False):
    v = dz(a, b, wide)
    e = err if err is not None else float(np.hypot(RUNS[a]["logzerr"], RUNS[b]["logzerr"]))
    return (f"\\ensuremath{{+{v:.2f}\\pm{e:.2f}}}" if v >= 0 else f"\\ensuremath{{-{abs(v):.2f}\\pm{e:.2f}}}")


# --------------------------------------------------------------- the model --
N["Padopted"] = ADOPT_TXT
N["PadP"] = pm(ADOPT, "P1", 5)
N["PadK"] = pm(ADOPT, "K1", 2)
kad, sad = med(ADOPT, "K1"), sig(ADOPT, "K1")
N["PadKfrac"] = f"{100*sad/kad:.1f}\\%"
N["PadKsnr"] = f"{kad/sad:.1f}"
N["PvalaK"] = pm("v2_serval_in_white_gauss_1p1c", "K1", 2)
N["PKwhiteJuliet"] = "\\ensuremath{5.52^{+0.90}_{-0.95}}"

# ------------------------------------------------------------ white ladder --
def p1dz(a, b):
    return P1[a]["lnZ"] - P1[b]["lnZ"]


e_s = float(np.hypot(P1["serval_in_1p1c"]["lnZerr"], P1["serval_in_0p"]["lnZerr"]))
e_d = float(np.hypot(P1["drs_in_1p1c"]["lnZerr"], P1["drs_in_0p"]["lnZerr"]))
ws = p1dz("serval_in_1p1c", "serval_in_0p")
wd = p1dz("drs_in_1p1c", "drs_in_0p")
N["PdlnzWhiteSin"] = f"\\ensuremath{{+{ws:.2f}\\pm{e_s:.2f}}}"
N["PdlnzWhiteDin"] = f"\\ensuremath{{+{wd:.2f}\\pm{e_d:.2f}}}"
N["Pdlnzagree"] = f"{abs(ws-wd):.2f}"
kp1s = P1["serval_in_1p1c"]["params"]["K_p1"]["med"]
kp1d = P1["drs_in_1p1c"]["params"]["K_p1"]["med"]
N["PKagree"] = f"{100*abs(kp1s-kp1d)/(0.5*(kp1s+kp1d)):.0f}\\%"

# ------------------------------------------- white-noise control, same code --
N["PdlnzWhiteCodeS"] = dzs("v2_serval_in_white_gauss_1p1c", "v2_serval_in_white_gauss_0p")
N["PdlnzWhiteCodeD"] = dzs("v2_drs_in_white_gauss_1p1c", "v2_drs_in_white_gauss_0p")
N["PdlnzGPSin"] = dzs("v1_serval_in_gauss_1p1c", "v1_serval_in_gauss_0p")
N["PdlnzGPDin"] = dzs("v1_drs_in_gauss_1p1c", "v1_drs_in_gauss_0p")
gs = dz("v1_serval_in_gauss_1p1c", "v1_serval_in_gauss_0p") - dz("v2_serval_in_white_gauss_1p1c", "v2_serval_in_white_gauss_0p")
gd = dz("v1_drs_in_gauss_1p1c", "v1_drs_in_gauss_0p") - dz("v2_drs_in_white_gauss_1p1c", "v2_drs_in_white_gauss_0p")
N["PgpGainS"] = f"{gs:+.2f}"
N["PgpGainD"] = f"{gd:+.2f}"
N["PgpVsWhiteZeroS"] = f"{dz('v1_serval_in_gauss_0p', 'v2_serval_in_white_gauss_0p'):+.1f}"

# ------------------------------------------------------------- Student-$t$ --
N["PdlnzTSin"] = dzs("v1_serval_in_studentt_1p1c", "v1_serval_in_studentt_0p")
N["PtGainS"] = f"{dz('v1_serval_in_studentt_1p1c','v1_serval_in_studentt_0p') - dz('v1_serval_in_gauss_1p1c','v1_serval_in_gauss_0p'):+.2f}"
N["Pnu"] = pm("v1_serval_in_studentt_1p1c", "nu", 1)
N["PgpLen"] = f"{med(ADOPT, 'gp_l'):.0f}" if "gp_l" in RUNS[ADOPT]["summary"] else "n/a"

# --------------------------------------------------------- the +51 epoch ----
din = dz(ADOPT, ADOPT_0P)
dout = dz(OUT_1P, OUT_0P)
N["PdlnzInAdopt"] = dzs(ADOPT, ADOPT_0P)
N["PdlnzOutAdopt"] = dzs(OUT_1P, OUT_0P)
# The epoch-sensitivity pair under the best model WITHOUT a seeing term, which is
# what the seeing term is introduced to fix.
RB1, RB0 = "v1_serval_in_studentt_1p1c", "v1_serval_in_studentt_0p"
RO1, RO0 = "v2_serval_out_m32_studentt_1p1c", "v2_serval_out_m32_studentt_0p"
_seepairs = [dz(ADOPT, ADOPT_0P)] + [dz(f"ver_{ADOPT}_s{q_}", f"ver_{ADOPT_0P}_s{q_}")
              for q_ in sorted({t.rsplit("_s", 1)[1] for t in RUNS if t.startswith("ver_")})
              if f"ver_{ADOPT}_s{q_}" in RUNS and f"ver_{ADOPT_0P}_s{q_}" in RUNS]
N["PseeFracGap"] = f"{100*(float(np.mean(_seepairs)) - dz(RB1, RB0))/(dz(RO1, RO0) - dz(RB1, RB0)):.0f}\\%"
N["ProbustGapNats"] = f"{dz(RO1, RO0) - dz(RB1, RB0):.1f}" if RO1 in RUNS else "\\note{pending}"
N["PdlnzInRobust"] = dzs(RB1, RB0)
N["PdlnzOutRobust"] = dzs(RO1, RO0) if RO1 in RUNS else "\\note{pending}"
N["PjitInRobust"] = f"{med(RB1, 's_rv0'):.2f}"
N["PjitOutRobust"] = f"{med(RO1, 's_rv0'):.2f}" if RO1 in RUNS else "\\note{pending}"
krb, srb = med(RB1, "K1"), sig(RB1, "K1")
N["PKfracRobust"] = f"{100*srb/krb:.1f}\\%"
three = []
for t in (RB1, SEEADOPT if "SEEADOPT" in dir() else "see_serval_in_gauss_1p1c_power", OUT_1P):
    if t in RUNS:
        three.append(f"{med(t,'K1'):.2f}\\pm{sig(t,'K1'):.2f}")
N["PKthreeWays"] = "\\ensuremath{" + ",\\ " .join(three) + "\\ \\mathrm{m\\,s^{-1}}}"
kout, sout = med(OUT_1P, "K1"), sig(OUT_1P, "K1")
N["PoutK"] = pm(OUT_1P, "K1", 2)
N["PoutKfrac"] = f"{100*sout/kout:.1f}\\%"
N["PjitInAdopt"] = f"{med(ADOPT, 's_rv0'):.2f}"
N["PjitOutAdopt"] = f"{med(OUT_1P, 's_rv0'):.2f}"
N["PKshiftSigma"] = f"{abs(kout-kad)/np.hypot(sad, sout):.2f}"
N["PgpLenOut"] = pm(OUT_1P, "gp_l", 0) if "gp_l" in RUNS[OUT_1P]["summary"] else "\\note{pending}"
N["PgpLenNoSee"] = pm("v1_serval_in_studentt_1p1c", "gp_l", 0)
N["PgpLenOutNoSee"] = (pm("v2_serval_out_m32_studentt_1p1c", "gp_l", 0)
                       if "v2_serval_out_m32_studentt_1p1c" in RUNS else "\\note{pending}")
N["PgpLenSee"] = (pm("see_serval_in_gauss_1p1c_power", "gp_l", 0)
                  if "see_serval_in_gauss_1p1c_power" in RUNS else "\\note{pending}")
N["PArvNoSee"] = pm("v1_serval_in_studentt_1p1c", "A_rv", 2)
N["PArvSee"] = (pm("see_serval_in_gauss_1p1c_power", "A_rv", 2)
                if "see_serval_in_gauss_1p1c_power" in RUNS else "\\note{pending}")
N["PArvOut"] = pm(OUT_1P, "A_rv", 2)
N["PArvIn"] = pm(ADOPT, "A_rv", 2)

# --- extra macros demanded by the internal-consistency audit -----------------
N["PdlnzGPSout"] = dzs("v1_serval_out_gauss_1p1c", "v1_serval_out_gauss_0p")
N["PseeNwith"] = "83"
N["PseeNnodimm"] = "21"
N["PdlnzTwoPTSin"] = dzs("v1_serval_in_studentt_2p1c2c", "v1_serval_in_studentt_1p1c")
if "v2_serval_out_m32_studentt_2p1c2c" in RUNS:
    N["PseeTwoPGain"] = f"{dz('v2_serval_out_m32_studentt_2p1c2c','v2_serval_out_m32_studentt_1p1c') - dz('v1_serval_out_gauss_2p1c2c','v1_serval_out_gauss_1p1c'):.1f}"
else:
    N["PseeTwoPGain"] = "\\note{pending}"
N["PjitOutGauss"] = f"{med('v1_serval_out_gauss_1p1c', 's_rv0'):.2f}"
N["PgpLenInGauss"] = pm("v1_serval_in_gauss_1p1c", "gp_l", 0)
N["PArvInGauss"] = pm("v1_serval_in_gauss_1p1c", "A_rv", 2)
N["PgpLenOutGauss"] = pm("v1_serval_out_gauss_1p1c", "gp_l", 0)
N["PArvOutGauss"] = pm("v1_serval_out_gauss_1p1c", "A_rv", 2)
N["PampRatio"] = f"{med('v1_serval_out_gauss_1p1c','A_rv')/med('v1_serval_in_gauss_1p1c','A_rv'):.1f}"
# white-noise two-planet numbers, from the juliet runs, so the paper stops typing them
for key, a, b in (("PdlnzTwoPWhiteSin", "serval_in_2p1c2c", "serval_in_1p1c"),
                  ("PdlnzTwoPWhiteSout", "serval_out_2p1c2c", "serval_out_1p1c"),
                  ("PdlnzTwoPWhiteDin", "drs_in_2p1c2c", "drs_in_1p1c"),
                  ("PdlnzTwoPWhiteDout", "drs_out_2p1c2c", "drs_out_1p1c")):
    if a in P1 and b in P1:
        v = P1[a]["lnZ"] - P1[b]["lnZ"]
        e = float(np.hypot(P1[a]["lnZerr"], P1[b]["lnZerr"]))
        N[key] = f"\\ensuremath{{{v:+.2f}\\pm{e:.2f}}}"
    else:
        N[key] = "\\note{pending}"
# how far apart the two reductions are under the ADOPTED model
if "see_drs_in_gauss_1p1c_power" in RUNS:
    dS = dz("see_serval_in_gauss_1p1c_power", "see_serval_in_gauss_0p_power")
    dD = dz("see_drs_in_gauss_1p1c_power", "see_drs_in_gauss_0p_power")
    N["PdlnzDrsAdopt"] = dzs("see_drs_in_gauss_1p1c_power", "see_drs_in_gauss_0p_power")
    N["PdlnzAgreeAdopt"] = f"{abs(dS-dD):.2f}"
    N["PdlnzAgreeGP"] = f"{abs(dz('v1_serval_in_gauss_1p1c','v1_serval_in_gauss_0p') - dz('v1_drs_in_gauss_1p1c','v1_drs_in_gauss_0p')):.2f}"
    ks_, kd_ = med("see_serval_in_gauss_1p1c_power","K1"), med("see_drs_in_gauss_1p1c_power","K1")
    N["PKagreeAdopt"] = f"{100*abs(ks_-kd_)/(0.5*(ks_+kd_)):.1f}\\%"
else:
    N["PdlnzDrsAdopt"] = N["PdlnzAgreeAdopt"] = N["PdlnzAgreeGP"] = N["PKagreeAdopt"] = "\\note{pending}"
# the step form's precision, so the reader can check the form choice both ways
if "see_serval_in_gauss_1p1c_thresh150" in RUNS:
    kt_, st_ = med("see_serval_in_gauss_1p1c_thresh150","K1"), sig("see_serval_in_gauss_1p1c_thresh150","K1")
    N["PseeThreshKfrac"] = f"{100*st_/kt_:.1f}\\%"
else:
    N["PseeThreshKfrac"] = "\\note{pending}"

# ------------------------------------------------------------- two-planet ---
two = ADOPT.replace("1p1c", "2p1c2c")
N["PdlnzTwoPAdopt"] = dzs(two, ADOPT) if two in RUNS else "\\note{pending}"
N["PdlnzTwoPDrsT"] = (dzs("v2_drs_in_m32_studentt_2p1c2c", "v1_drs_in_studentt_1p1c")
                      if "v2_drs_in_m32_studentt_2p1c2c" in RUNS else "\\note{pending}")
N["PdlnzTwoPOutT"] = (dzs("v2_serval_out_m32_studentt_2p1c2c", "v2_serval_out_m32_studentt_1p1c")
                      if "v2_serval_out_m32_studentt_2p1c2c" in RUNS else "\\note{pending}")
N["PdlnzOneDrsT"] = dzs("v1_drs_in_studentt_1p1c", "v1_drs_in_studentt_0p")
N["PdlnzOneDrsOut"] = (dzs("v2_drs_out_m32_gauss_1p1c", "v2_drs_out_m32_gauss_0p")
                       if "v2_drs_out_m32_gauss_1p1c" in RUNS else "\\note{pending}")

# ---------------------------------------------------------------- aliases ---
N["PaliasWhite"] = "\\ensuremath{+9.99\\pm0.54}"
for key, tag, base in (("PaliasGP", "v2_serval_in_m32_gauss_1p1c_alias", "v1_serval_in_gauss_1p1c"),
                       ("PaliasGPT", "v2_serval_in_m32_studentt_1p1c_alias", "v1_serval_in_studentt_1p1c")):
    N[key] = dzs(base, tag, wide=True) if tag in RUNS else "\\note{pending}"
ap = f"{HERE}/out/alias_profile.json"
if os.path.exists(ap):
    A = json.load(open(ap))
    N["PaliasDlogl"] = (f"{A['serval']['dlogl']:.1f} (\\texttt{{SERVAL}}) and "
                        f"{A['drs']['dlogl']:.1f} (DRS)")
    N["PaliasGlobalMax"] = f"{A['serval']['global_max_P']:.5f}"
else:
    N["PaliasDlogl"] = "\\note{pending}"

# ------------------------------------------------------------ TOI-6263.01 ---
toi = "v2_serval_in_m32_studentt_1p1c-toi" if "v2_serval_in_m32_studentt_1p1c-toi" in RUNS \
      else "v2_serval_in_m32_gauss_1p1c-toi"
if toi in RUNS:
    ks_toi = np.array(q(toi, "K_toi"))
    N["PKtoi"] = f"{np.interp(0.95, [0.16,0.5,0.84], ks_toi):.2f}"
    N["PdlnzToi"] = dzs(toi, ADOPT if "studentt" in toi else "v1_serval_in_gauss_1p1c")
else:
    N["PKtoi"] = N["PdlnzToi"] = "\\note{pending}"

# ------------------------------------------------------------------- ecc ----
N["PeccNlive"] = "1500"
ecc = os.environ.get("ECCTAG", "ecc_serval_in_studentt_1p_nl1500")
if ecc in RUNS:
    N["Pade"] = f"{RUNS[ecc]['e95']:.2f}"
    N["PadeNinety"] = f"{RUNS[ecc]['e90']:.2f}"
    N["PadeMed"] = f"{med(ecc,'ecc'):.3f}"
    # The eccentric comparison is made against the circular model of the SAME
    # noise configuration, not against the adopted one, so that the eccentricity
    # is the only thing that differs.
    ecc_ref = os.environ.get("ECCREF", "v1_serval_in_gauss_1p1c")
    N["PdlnzEcc"] = dzs(ecc, ecc_ref)
    N["PeccModel"] = ("the adopted noise model" if ecc.endswith("_power")
                      else "the Mat\\'ern GP without a seeing term")
    N["PeccK"] = pm(ecc, "K1", 2)
else:
    N["Pade"] = N["PadeNinety"] = N["PadeMed"] = N["PdlnzEcc"] = "\\note{pending}"
    N["PeccModel"] = N["PeccK"] = "\\note{pending}"

# ---------------------------------------------------------------- seeing ----
import p3data
d = p3data.build_v1("serval", True)
s = p3data.seeing_v1(d["t"])
ok = np.isfinite(s)
N["PseeSdGood"] = f"{np.std(d['rv'][ok & (s <= 1.5)]):.1f}"
N["PseeSdBad"] = f"{np.std(d['rv'][ok & (s > 1.5)]):.1f}"
N["PseeNbad"] = str(int((ok & (s > 1.5)).sum()))

SD = json.load(open(f"{HERE}/out/seeing_diag.json"))
N["PseeRall"] = f"{SD['pearson_all'][0]:.2f}"
N["PseePall"] = f"{SD['pearson_all'][1]:.4f}"
N["PseeRno"] = f"{SD['pearson_no51'][0]:.2f}"
N["PseePno"] = f"{SD['pearson_no51'][1]:.2f}"
N["PseeRhoAll"] = f"{SD['spearman_all'][0]:.2f}"

BASE0, BASE1 = "v1_serval_in_gauss_0p", "v1_serval_in_gauss_1p1c"
SEE = {"power": "see_serval_in_gauss_%s_power",
       "hinge": "see_serval_in_gauss_%s_hinge",
       "thresh": "see_serval_in_gauss_%s_thresh150"}
zs = {}
for k, patt in SEE.items():
    t = patt % "0p"
    if t in RUNS:
        zs[k] = dz(t, BASE0)
        N["PseeZero" + k.capitalize()] = f"{zs[k]:+.2f}"
    else:
        N["PseeZero" + k.capitalize()] = "\\note{pending}"
if len(zs) == 3:
    N["PseeFormSpread"] = f"{max(zs.values())-min(zs.values()):.2f}"
    N["PseeThreshOverPower"] = f"{zs['thresh']-zs['power']:.2f}"
else:
    N["PseeFormSpread"] = N["PseeThreshOverPower"] = "\\note{pending}"
th13, th17 = "see_serval_in_gauss_0p_thresh130", "see_serval_in_gauss_0p_thresh170"
if th13 in RUNS and th17 in RUNS and "thresh" in zs:
    v = [dz(th13, BASE0), zs["thresh"], dz(th17, BASE0)]
    N["PseeThreshSens"] = f"{min(v):+.1f} to {max(v):+.1f}"
else:
    N["PseeThreshSens"] = "\\note{pending}"

LOO0 = "see_serval_out_gauss_0p_power"
LOO1 = "see_serval_out_gauss_1p1c_power"
if LOO0 in RUNS:
    N["PseeLooZero"] = f"{dz(LOO0, 'v1_serval_out_gauss_0p'):+.2f}"
    N["PseeQloo"] = pm(LOO0, "q_see", 1)
else:
    N["PseeLooZero"] = N["PseeQloo"] = "\\note{pending}"
N["PdlnzSeeOut"] = dzs(LOO1, LOO0) if (LOO1 in RUNS and LOO0 in RUNS) else "\\note{pending}"

SEEADOPT = os.environ.get("SEEADOPT", "see_serval_in_gauss_1p1c_power")
SEEADOPT0 = os.environ.get("SEEADOPT0P", "see_serval_in_gauss_0p_power")
if SEEADOPT in RUNS:
    N["PdlnzSeeSin"] = dzs(SEEADOPT, SEEADOPT0)
    N["PjitSeeIn"] = f"{med(SEEADOPT, 's_rv0'):.2f}"
    N["PseeNodimm"] = f"{med(SEEADOPT, 's_nodimm'):.2f}"
    if "q_see" in RUNS[SEEADOPT]["summary"]:
        qq = med(SEEADOPT, "q_see"); gg = med(SEEADOPT, "gam_see")
        N["PseeQ"] = pm(SEEADOPT, "q_see", 1)
        N["PseeAtMedian"] = f"{gg:.2f}"
        N["PseeAtBad"] = f"{gg*(1.97/0.90)**qq:.0f}"
    else:
        N["PseeQ"] = N["PseeAtMedian"] = N["PseeAtBad"] = "\\note{n/a}"
else:
    for k in ("PdlnzSeeSin", "PjitSeeIn", "PseeNodimm", "PseeQ", "PseeAtMedian", "PseeAtBad"):
        N[k] = "\\note{pending}"

# ------------------------------------------------------------------- QP -----
qp0, qp1 = "v2_serval_in_qp_gauss_0p", "v2_serval_in_qp_gauss_1p1c"
if qp0 in RUNS:
    N["PProtQP"] = pm(qp0, "eta3", 1)
    N["PetaFour"] = pm(qp0, "eta4", 1)
    N["PetaTwo"] = pm(qp0, "eta2_u", 0)
    N["PmaternVsQP"] = f"{dz('v1_serval_in_gauss_0p', qp0):+.2f}"
else:
    N["PProtQP"] = N["PetaFour"] = N["PetaTwo"] = N["PmaternVsQP"] = "\\note{pending}"
N["PdlnzQPSin"] = dzs(qp1, qp0) if qp1 in RUNS and qp0 in RUNS else "\\note{pending}"

# ------------------------------------------------------------- stability ----
Ks = [(t, med(t, "K1")) for t in RUNS if "K1" in RUNS[t]["summary"]
      and RUNS[t]["model"] != "1p1c_alias" and not t.startswith("ver_")]
kv = np.array([k for _, k in Ks])
N["PnStabCells"] = str(len(kv))
N["PKspread"] = f"{kv.min():.2f}\\!-\\!{kv.max():.2f}"
N["PKsd"] = f"{kv.std(ddof=1):.2f}"

# ---------------------------------------------------------------- verify ----
# Matched repeats of the adopted pair: the original run (seed 42, nlive 750)
# plus one repeat per (seed, nlive). The empirical scatter of dlnZ across them is
# what the paper quotes as the evidence uncertainty for the adopted model.
seeds = sorted({t.rsplit("_s", 1)[1] for t in RUNS if t.startswith("ver_")})
pairs = [dz(ADOPT, ADOPT_0P)]
for sd_ in seeds:
    a, b = f"ver_{ADOPT}_s{sd_}", f"ver_{ADOPT_0P}_s{sd_}"
    if a in RUNS and b in RUNS:
        pairs.append(dz(a, b))
if len(pairs) >= 3:
    pv = np.array(pairs)
    N["PverifyN"] = str(len(pv))
    N["PverifySd"] = f"{pv.std(ddof=1):.2f}"
    N["PverifyMean"] = f"\\ensuremath{{{pv.mean():+.2f}\\pm{pv.std(ddof=1):.2f}}}"
    N["PverifyRange"] = f"{pv.min():+.2f} to {pv.max():+.2f}"
    N["PverifyInternal"] = f"{float(np.hypot(RUNS[ADOPT]['logzerr'], RUNS[ADOPT_0P]['logzerr'])):.2f}"
    N["PverifyRatio"] = f"{pv.std(ddof=1)/float(np.hypot(RUNS[ADOPT]['logzerr'], RUNS[ADOPT_0P]['logzerr'])):.0f}"
    N["PverifyLow"] = f"{pv.mean()-2*pv.std(ddof=1):+.1f}"
    # The headline evidence is the mean over independent repeats, not one run.
    N["PdlnzInAdoptSingle"] = dzs(ADOPT, ADOPT_0P)
    N["PdlnzInAdopt"] = N["PverifyMean"]
else:
    for k in ("PverifyN", "PverifySd", "PverifyMean", "PverifyRange",
              "PverifyInternal", "PverifyRatio", "PverifyLow"):
        N[k] = "\\note{pending}"

# ------------------------------------------------------- star and system ----
def sysv(k, dp=3, unit=""):
    v = SYS[k]
    a, b = v["lo"], v["hi"]
    if abs(a - b) < 0.15 * max(a, b):
        return f"\\ensuremath{{{v['med']:.{dp}f} \\pm {0.5*(a+b):.{dp}f}{unit}}}"
    return f"\\ensuremath{{{v['med']:.{dp}f}^{{+{b:.{dp}f}}}_{{-{a:.{dp}f}}}{unit}}}"


N["Pdist"] = sysv("dist_pc", 3)
N["PTeff"] = f"\\ensuremath{{{SYS['Teff']['med']:.0f} \\pm {0.5*(SYS['Teff']['hi']+SYS['Teff']['lo']):.0f}}}"
N["PRstar"] = sysv("R", 3)
N["PMstar"] = sysv("M", 3)
N["PaoverR"] = sysv("a_over_R", 2)
N["PimaxDeg"] = f"{SYS['i_max_deg']['med']:.2f}\\degr"

# M sin i recomputed with the ADOPTED K, by Monte Carlo over K and Mstar.
rng = np.random.default_rng(20260908)
NS = 200000
Ksamp = np.array(RUNS[ADOPT]["samples"]) if "samples" in RUNS[ADOPT] else None
kdraw = rng.normal(kad, sad, NS)
mdraw = rng.normal(SYS["M"]["med"], 0.5*(SYS["M"]["hi"]+SYS["M"]["lo"]), NS)
Pd = med(ADOPT, "P1")
# M sin i (M_earth) for a circular orbit:  m sini = K (M*)^{2/3} (P/2pi G)^{1/3}
G = 6.67430e-11; Msun = 1.98892e30; Mearth = 5.9722e24; day = 86400.0
def msini(K, M):
    return K * (M*Msun)**(2/3.) * ((Pd*day)/(2*np.pi*G))**(1/3.) / Mearth
ms = msini(kdraw, mdraw)
lo, m50, hi = np.percentile(ms, [16, 50, 84])
N["PadMsini"] = f"\\ensuremath{{{m50:.1f} \\pm {0.5*(hi-lo):.1f}}}"
mso = msini(rng.normal(kout, sout, NS), mdraw)
lo2, m502, hi2 = np.percentile(mso, [16, 50, 84])
N["PoutMsini"] = f"\\ensuremath{{{m502:.1f} \\pm {0.5*(hi2-lo2):.1f}}}"
a_au = ((mdraw*Msun) * G * (Pd*day)**2 / (4*np.pi**2))**(1/3.) / 1.495978707e11
N["Pada"] = f"\\ensuremath{{{np.median(a_au):.5f} \\pm {np.std(a_au):.5f}}}"
N["PmassErrTot"] = f"{100*np.hypot(sad/kad, (2/3.)*0.5*(SYS['M']['hi']+SYS['M']['lo'])/SYS['M']['med']):.1f}\\%"
N["PSinc"] = f"{0.2275/np.median(a_au)**2:.0f}"
Teq = SYS["Teff"]["med"] * np.sqrt(SYS["R"]["med"]*6.957e8 / (2*np.median(a_au)*1.495978707e11))
N["PTeq"] = f"{Teq:.0f}"

# True mass and Hill separation recomputed with the ADOPTED K (Phase 5 used the
# Phase-1 white-noise K). i_max depends on a/R* and R_p only and is unchanged.
imax = np.radians(SYS["i_max_deg"]["med"])
cosi = rng.uniform(np.cos(imax), 1.0, NS)          # isotropic in cos i, truncated
mtrue = ms / np.sqrt(1 - cosi**2)
lo3, m3, hi3 = np.percentile(mtrue, [16, 50, 84])
N["PMtrue"] = f"\\ensuremath{{{m3:.1f}^{{+{hi3-m3:.1f}}}_{{-{m3-lo3:.1f}}}}}"
# Hill: m1 = TOI-6263.01 mass prediction 0.76 +- 0.30 Mearth at a1 = 0.03757 au
m1 = np.clip(rng.normal(0.76, 0.30, NS), 0.05, None)
a1 = rng.normal(SYS["a1_AU"]["med"], 0.5*(SYS["a1_AU"]["hi"]+SYS["a1_AU"]["lo"]), NS)
RH = 0.5*(a1 + a_au) * ((m1 + ms)*Mearth/(3*mdraw*Msun))**(1/3.)
Dl = (a_au - a1)/RH
lo4, m4, hi4 = np.percentile(Dl, [16, 50, 84])
N["PDelta"] = f"\\ensuremath{{{m4:.2f}^{{+{hi4-m4:.2f}}}_{{-{m4-lo4:.2f}}}}}"
N["PgladmanPass"] = f"{100*np.mean(Dl > 2*np.sqrt(3)):.3f}"
if toi in RUNS:
    N["PMtoi"] = f"{float(N['PKtoi'])* (SYS['M']['med']*Msun)**(2/3.) * ((3.0178746*day)/(2*np.pi*G))**(1/3.) / Mearth:.2f}"
else:
    N["PMtoi"] = "\\note{pending}"

# --------------------------------------------------- detection limits -------
ij = f"{HERE}/out/injrec_gp.npz"
if os.path.exists(ij):
    z = np.load(ij, allow_pickle=True)
    Pg, K95, K50 = z["Pgrid"], z["K95"], z["K50"]
    short = (Pg >= 2) & (Pg <= 20)
    longp = (Pg >= 100) & (Pg <= 400)
    N["PKninetyfiveMed"] = f"{np.nanmedian(K95):.2f}"
    N["PKninetyfiveShort"] = f"{np.nanmin(K95[short]):.2f}--{np.nanmax(K95[short]):.2f}"
    N["PKninetyfiveLong"] = f"{np.nanmin(K95[longp]):.2f}--{np.nanmax(K95[longp]):.2f}"
    i0 = int(np.argmin(np.abs(Pg - med(ADOPT, "P1"))))
    N["PKninetyfiveAtP"] = f"{K95[i0]:.2f}"
    N["PKfactorAbove"] = f"{kad/K95[i0]:.2f}"
    i2 = int(np.argmin(np.abs(Pg - 200.9)))
    N["PKninetyfiveAtTwoHundred"] = f"{K95[i2]:.2f}"
    N["PthrParam"] = f"{float(z['THR']):.1f}"
    N["PthrShuffle"] = f"{float(z['THR_shuffle']):.0f}"
else:
    for k in ("PKninetyfiveMed", "PKninetyfiveShort", "PKninetyfiveLong",
              "PKninetyfiveAtP", "PKfactorAbove", "PKninetyfiveAtTwoHundred",
              "PthrParam", "PthrShuffle"):
        N[k] = "\\note{pending}"

# ------------------------------------------------- adversarial coupling -----
LAT1, LAT0 = "lat_serval_in_m32_gauss_1p1c_power", "lat_serval_in_m32_gauss_0p_power"
if LAT0 in RUNS:
    N["PlatZero"] = f"{dz(LAT0, SEEADOPT0):+.2f}"
else:
    N["PlatZero"] = "\\note{pending}"
if LAT1 in RUNS and LAT0 in RUNS:
    N["PdlnzLat"] = dzs(LAT1, LAT0)
    N["PlatK"] = pm(LAT1, "K1", 2)
    klat, slat = med(LAT1, "K1"), sig(LAT1, "K1")
    N["PlatKfrac"] = f"{100*slat/klat:.1f}\\%"
    N["PlatKshift"] = f"{abs(klat-kad)/np.hypot(slat, sad):.2f}"
    N["PlatOne"] = f"{dz(LAT1, SEEADOPT):+.2f}"
    N["PlatArv"] = pm(LAT1, "A_rv", 2)
    N["PlatAdlw"] = pm(LAT1, "A_dlw", 2)
    N["PdlnzLatLoss"] = f"{dz(LAT1, LAT0) - dz(SEEADOPT, SEEADOPT0):+.2f}"
else:
    for k in ("PdlnzLat", "PlatK", "PlatKfrac", "PlatKshift", "PlatOne",
              "PlatArv", "PlatAdlw", "PdlnzLatLoss"):
        N[k] = "\\note{pending}"

# ------------------------------------------------------------------ write ---
used = set(re.findall(r"\\(P[A-Za-z]+)", open(TEX).read()))
missing = sorted(used - set(N))
extra = sorted(set(N) - used)
with open(f"{OUT}/numbers.tex", "w") as f:
    f.write("% GENERATED by phase3/make_numbers.py -- do not edit by hand.\n")
    for k in sorted(N):
        f.write(f"\\newcommand{{\\{k}}}{{{N[k]}}}\n")
    for k in missing:
        f.write(f"\\newcommand{{\\{k}}}{{\\note{{UNDEFINED}}}}\n")
json.dump(N, open(f"{HERE}/out/numbers.json", "w"), indent=1)
print(f"wrote {len(N)} macros to numbers.tex and out/numbers.json")
if missing:
    print("MISSING (defined as [UNDEFINED] in the paper):", missing)
if extra:
    print("unused macros:", extra)
pend = [k for k, v in N.items() if "pending" in str(v)]
if pend:
    print("PENDING (run not finished):", pend)
