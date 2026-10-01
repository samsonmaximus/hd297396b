# Phase 3 (final) — correlated noise, a robust likelihood, and a seeing term

HD 297396 (TOI-6263). One log-likelihood over the stacked `[RV, dLW]` vector:
per-label offsets and jitter for both series, error-scaling β, and a GP whose
hyperparameters are shared between the two series with separate amplitudes (the
Rajpaul et al. 2015 / Lillo-Box et al. 2021 structure), optionally with a
multivariate Student-t likelihood, a rank-1 shared-latent coupling, or a
seeing-dependent RV variance term. Evidences from `dynesty` nested sampling.

Everything is run on **`data-v1`**, the Phase-1 freeze, via `p3data.build_v1()`.
The Phase-3-native 4σ-clipped builder (`p3data.build()`) is retained only so the
older runs can be reproduced; nothing in the paper uses it.

## The rule that governs this directory

**No number in the paper or the report is typed by hand.** They are generated:

```
out/*.json                 <- one JSON per runner (ladder_v2, seeing, ecc_v1, ...)
    |  assemble.py
out/phase3_final.json      <- one merged, normalised table of every run
    |  make_numbers.py -> ../phase6/overleaf/numbers.tex + out/numbers.json
    |  make_tables.py  -> ../phase6/overleaf/tab_*.tex
    |  make_report.py  -> phase3_report.md
    |  figs_final.py   -> ../phase6/overleaf/fig_*.png
```

`./refresh.sh` runs the first three in order. `make_numbers.py` asserts that the
set of macros it defines is exactly the set the paper uses, and emits
`\note{UNDEFINED}` for any macro the paper cites that no run produced — so a red
UNDEFINED in the compiled PDF means a number is being quoted that nothing
computed. `adopt.env` names the adopted (headline) run; change it there and
`./refresh.sh` re-derives the whole paper against a different row.

## Runners

| file | what it runs |
|---|---|
| `ladder_v2.py` | step 1: the 18 rungs that existed only on the clipped set — white-noise control through this code, alias, +TOI, the QP branch, DRS Student-t, both `out` columns |
| `seeing.py` | step 2: the seeing-dependent noise term. Forms, priors and read-out fixed in `PREREGISTRATION_step2.md` **before** the first run |
| `seeing_diag.py` | the diagnostics that say what the seeing predictor is and is not supported by |
| `adversarial.py` | step 3: rank-1 shared-latent coupling on the adopted noise model |
| `ecc_v1.py`, `ecc_see.py` | eccentric cells at nlive 1500 (the project's own floor is 1000; the pre-existing `data-v1` eccentric run was at 750) |
| `verify_v1.py` | run-to-run evidence scatter of the adopted pair across seeds and live-point counts |
| `injrec_gp.py` | step 5.1: detection limits in the whitened frame of the adopted covariance |
| `alias_profile.py` | profile-likelihood period scan, amplitude/phase/offsets profiled out analytically |
| `runq*.py` | sequential controllers; each step appends to `logs/queue.log` |

`ladder.py`, `stability.py`, `pscan*.py`, `val_*.py`, `p3mcmc.py`, `plots.py`,
`report.py`, `fig_v1.py` are the earlier, clipped-set versions. They are kept for
provenance and are not on the path to any published number.

## Reproduce

```
pip install numpy scipy dynesty astropy matplotlib pandas
python3 ladder_v2.py            # ~18 runs, ~2 h on 2 cores
python3 seeing.py stage1        # 6 runs
python3 seeing.py out/seeing_stage2.json
python3 ecc_v1.py ; python3 ecc_see.py
python3 verify_v1.py
python3 injrec_gp.py
python3 adversarial.py
./refresh.sh ; python3 figs_final.py ; python3 make_report.py
```

Runs are cached in `out/ns_<tag>.pkl` keyed on the tag alone, so **a new data
selection under an old tag silently returns the old result**. Every new
configuration in this directory gets a new tag prefix (`v2_`, `see_`, `lat_`,
`ecc_`, `ver_`).

## Five things to know before trusting anything here

1. **The GP is worth about one nat.** Compared against white noise run through
   *this same code*, the Matérn GP buys +1.10 (SERVAL) and +0.24 (DRS) on the
   planet evidence. The +2.4 implied by comparing against Phase 1's `juliet`
   runs was mostly the change of code. Two implementations of the same model
   differ by ~1 nat; do not interpret smaller differences.
2. **`dynesty`'s `logzerr` understates the scatter, badly on the seeing model.**
   Four repeats of the adopted pair span 13.8–17.6 in ΔlnZ (sd 1.7) against an
   internal estimate of 0.39. The paper quotes the mean of repeats.
3. **The GP length-scale posterior is bimodal** (~10³ d and ~70–110 d) and the
   +51 epoch selects the mode. Evidences of runs in different modes are not
   comparable; two such comparisons are declined in the paper for that reason.
4. **The seeing term's leverage comes from the epoch it excuses.** Refitted with
   that epoch removed it buys +1.05 nats instead of +19.94. It is a
   pre-registered, externally keyed way to down-weight one epoch — not a
   demonstrated seeing–jitter law. `seeing_diag.py` prints the evidence.
5. **emcee fails silently on this posterior** (K biased low by up to 2.3σ on
   injected data that nested sampling recovers unbiased). `p3mcmc.py` documents
   the failure and is used for nothing else.
