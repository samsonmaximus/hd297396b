# HD 297396 b, v12 — reproduction scripts

Inputs (put in `data/`):
- `HD297396_rvbank_full.csv`: the 108 HARPS-RVBank rows for HD 297396 (Perdelwitz et al. 2024, corrected release 2026-04-20).
- `table4.dat.gz`: the full HARPS-RVBank table (used only by `xstar.py` and `counts.py`).

Run from this folder. Each script prints the numbers it produces.

| Script | Produces (paper section) |
|---|---|
| `core.py` | Five rejection criteria, nightly binning, the 104-epoch set (Sect. 2.3) |
| `pg.py`, `split.py` | Periodogram with per-label offsets and jitters; subset fits (Sects. 4.2, 5.4) |
| `fap.py` | Bootstrap global FAPs, 10 000 draws, 104 and 103 epochs (Sect. 4.2) |
| `counts.py` | 1214 searchable stars (Sect. 2.2; 1221 before the 2026-09-30 night-boundary fix) |
| `alias.py` | 1.301-d alias simulations (Sect. 4.4) |
| `heldout.py` | Held-out test across programme groups (Sect. 5.4) |
| `xstar.py` | Cross-star control, 611 stars (Sect. 5.5; 612 before the 2026-09-30 night-boundary fix) |
| `fwd2.py` | Forward model of coherent rotation (Sect. 5.3) |
| `k90.py` | 90% recovery amplitude in 25–40 d (Sect. 5.6) |
| `wn_checks.py` | Jackknife, alias deletions, 200.9-d signal, TOI-6263.01 limit, harmonic bands (Sects. 4.4, 5.6, 6, 7) |
| `limits.py` | Detection limits (Sect. 5.8) |
| `indic.py` | Activity-indicator periodograms and correlations (Sect. 5.1, Fig. 4, Table 5) |
| `gpdata.py`, `gpns.py` | Independent implementation of the adopted Matérn GP, nested sampling (App. A) |
| `figs_a.py`, `figs_b.py`, `figs_rv.py`, `style.py` | Figures |

The primary model ladder (Table 3) comes from the phase-3 pipeline (`phase3/`), not from these scripts.

## Added in v13 (2026-09-25)

Run `gpdata.py` first; it writes `frozen104.csv`, which the GP scripts read.

| Script | Produces (v13 section) | Output |
|---|---|---|
| `baluev.py` | Baluev FAPs, 104/103 epochs (4.2) | printout |
| `gpmap.py`, `gpfap.py` | ML hyperparameters of the adopted Matérn model; covariance-weighted periodogram and its FAP (4.2) | `gpmap.json`, `gpfap.json` |
| `gp_ul.py`, `gp_typical.py` | posterior velocity-amplitude limits (Matérn, QP); periodogram under posterior-typical covariances (4.2) | `gp_ul.json`, `gp_typical.json` |
| `qpgp.py` | quasi-periodic GP with P_rot free over 10–100 d (4.2) | `qpgp.json` |
| `qp_profile.py`, `qp_split.py`, `qp_null.py` | profile likelihood of the dLW quasi-period, subsets, H-alpha, window; aperiodic null (3.2) | `qp_profile.json`, `qp_split.json`, `qp_null_3.json` |
| `growth.py`, `blocks.py`, `figs_coh.py` | signal growth with N; four-block fits; Fig. 4 (5.2) | `growth.json`, `blocks_104.npy` |
| `blockact.py` | block amplitude vs activity (5.2) | printout |
| `fastcheck.py`, `semian.py`, `semian2.py` | semi-analytic evidences at fixed jitters (method checks; the fixed-jitter FIP of 4.3) | `semian.json`, `semian2_20_10.json` |
| `fip.py`, `fip_seed.py`, `fip_extra.py` | full-band evidences and FIP with all noise marginalised; seed check; third-planet check and peak-window fraction (4.3, App. A) | `fip_seed11.json`, `fip_seed23.json`, `fip_extra.json` |
| `aliasband.py`, `aliasband_marg.py` | alias comparison at fixed / marginalised jitters (4.4) | `aliasband*.json` |
| `rot2240.py`, `rot1745.py` | rotation-band robustness 22–40 and 17–45 d (5.3, 5.6) | `rot2240.json`, `rot1745.json` |
| `fwd_qp.py`, `fwd_qp_long.py` | evolving-spot (quasi-periodic) forward models (5.3) | `fwd_qp.json`, `fwd_qp_long.json` |
| `ecc.py`, `ecc_check.py`, `ecc_check2.py`, `ecc_ns.py` | eccentricity MCMC; the periastron-spike solution; (nested-sampling check, not completed) (4.5) | `ecc.json` |
| `nonzp.py` | periodogram before the nightly zero-point correction; discrepant-night σ (2.3, 5.5) | printout |
| `heldout_new.py` | the out-of-sample contrast test on the 2022–2024 HARPS spectra (8.3); see `../HELDOUT_TEST.md` | printout |
