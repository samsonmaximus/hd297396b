# Where every number in the paper comes from (v12 base, with v13, v14 and v15 additions)

Two sources. **A** = the scripts in `analysis/` (written for v12, run on 2026-09-22 on the RVBank rows).
**P** = the existing project pipeline (phase folders and v10/v11 macro files), carried over without change.
Any number that depended on the dropped seeing-term model was recomputed under A or removed.

| Quantity | Value | Source |
|---|---|---|
| 104 / 103 epochs; rejections C2 = 1, C4 = 1 | | A `core.py` (reproduces P exactly) |
| Periodogram peak, Δχ² 32.3 / 44.0; DRS 27.9 / 37.5 | | A `pg.py` |
| Bootstrap FAP 1.4e-3 (14/10 000); 0/10 000; ≈3e-6 extrapolated | | A `fap.py` |
| Baluev bound 4.9e-5 | | P (v11) |
| 1214 searchable stars (1221 before the night-boundary correction of 2026-09-30) | | A `counts.py` |
| 072.C-0488 jitter 7.7 / 3.1 m/s | | A `wn_checks.py` / `split.py` |
| ΔlnZ ladder, all K in Table 3, M sin i under other treatments | | P `phase3/out/*.json`, `v10/numbers*.tex` |
| Adopted P = 4.26837 ± 0.00027, K = 5.52 ± 0.84, T_conj | | P `phase3_final.json`, run `v1_serval_in_gauss_1p1c` |
| Independent GP: ΔlnZ +7.33 ± 0.35, K 5.18 ± 0.81 | | A `gpns.py` |
| Full-band correction 5.34 nats | | arithmetic, ln[ln(1000/1.05)/ln(4.34/4.20)] |
| M sin i 11.8 ± 1.9; M_p 14.5 +8.5 −3.1 | | A (Monte Carlo from K, M★, TESS inclination cut) |
| Alias matched-prior +6.02 / +9.99; window percentiles; parent periods | | P |
| Alias simulation 0/4000, 96% | | A `alias.py` |
| Alias wins all 104 deletions (min Δ 5.3) | | A `wn_checks.py` |
| Four-block coherence (52/26/14/12; 15°; p = 0.61, 0.79) | | A `split.py` block fit |
| τ > 2300 d | | A (profile likelihood, white-noise frame) |
| Forward model 0/20 000 at 4.5 m/s; 0.4% at 12; 2.4% at 15; Keplerian 72% | | A `fwd2.py` |
| Held-out: 4.2692 ± 0.0010; Δχ² 17.6; p 1.6e-3 / 3e-4; reverse 28.4; 5.1975-d peak | | A `heldout.py` |
| Cross-star: 611 stars, median 1.21, none > 13.8, 54 stars max 8.1 (before the night-boundary correction of 2026-09-30: 612, 1.24, 49, 7.6); NZP 0.18 m/s | | A `xstar.py` (`xstar.csv`; kept = control median < 3), `wn_checks.py` |
| Harmonic bands 15.9 / 11.0 / 10.1 vs 20.1 / 21.6 / 21.9; K90 = 4.5 m/s | | A `wn_checks.py`, `k90.py` |
| Indicator table and figure | | A `indic.py` |
| Rotation: Noyes values, seasonal stacks, QP hyperparameter, 0.03 dex sensitivity | | P |
| v sin i exclusion of P_rot = P0 | | arithmetic from Table 2 |
| TESS (63 ppm, 96% retention, Rayleigh p), ASAS-SN (all), companions (all) | | P |
| Detection limits (K95 5–6 m/s, 4.6 at P0) | | A `limits.py` |
| 200.9-d: ΔlnZ +5.98, window 0.25 elements, BERV Δχ² 43, r = −0.14 | | P |
| 200.9-d: Δχ² 31.6, FAP 0.2%, K2 4.9 ± 0.6, 0.25σ shift of K_b | | A `wn_checks.py` |
| TOI-6263.01: K 1.3, < 2.7 m/s; Hill 9.2 | | A / P |
| Jackknife 0.10 rms, 0.44 max; whitened residual 6.0σ, 2.46 | | A `wn_checks.py`, `gpns.py` posterior |
| Held-out HARPS epochs: phases ≈ 0.0, 0.75, 0.0; 5.5 ± 1.0 m/s; ≈1.2σ | | A (approximate times; recompute with header BJDs) |

## New or changed in v13 (source **B** = scripts added to `analysis/` on 2026-09-25, run on the same RVBank rows)

| Quantity | Value | Source |
|---|---|---|
| C2 flags 2 spectra (2012-12-27, G2 mask; 2013-06-23, also fails C4); 2 rejected in total | | A `core.py` (printout `rejected C2 2 C3 0 C4 1 kept 106`) |
| Discrepant night: indicator z-scores within 1.2 robust σ; NZP 0.10 m/s; previous night −0.6 m/s | | inline check on `HD297396_rvbank_full.csv` (rows 58–59) |
| Noyes τc = 23.7 d, Ro = 1.21, P_rot = 28.6 d (22.7–36.0 d); MH08 27.6 d; Ro(P0) = 0.18 → log R'HK ≈ −4.3 | | arithmetic (Noyes 1984 eq. 4 and R'HK–Ro fit; Mamajek & Hillenbrand 2008 eq. 5), B−V = 1.09. **v12 value 34.1 d was 8 × P0 pasted in error**; `phase9/kinematics/kinematics.json` has 23.6 d |
| Baluev FAP 2.2e-3 (104), 6.7e-6 (103); archive-wide 1.7 (bootstrap, 104) and 8e-3 (Baluev, 103) | | B `baluev.py` (**v12's 4.9e-5 was wrong**) |
| GP-weighted periodogram: ML velocity GP amplitude 0.1 m/s (prior floor); Δχ² 32.4 / 44.0; FAP 12/10⁴ | | B `gpmap.py` → `gpmap.json`; `gpfap.py` → `gpfap.json` |
| Signal growth: inside 90 % band at 90/90 and 89/89 steps; conformity p 0.88 / 0.83; night step +7.7 | | B `growth.py` → `growth.json`; figure `figs_coh.py` |
| Four blocks (recomputed): K 5.8/6.3/3.7/5.8, phase rms 15.4°, p 0.61 / 0.79 (3 dof); global K 5.72 ± 0.81 used for the band in Fig. 4 | | B `blocks.py` → `blocks_104.npy` (v12's figure used 5.75 ± 0.58, the 103-epoch value) |
| Block K vs log R'HK slope −10 ± 19 m/s per dex, span 0.16 dex | | B `blockact.py` |
| Alias, marginalised, equal-log-width Jeffreys bands: +5.7 (104), +9.9 (103); frequency-neutral 4.5 / 8.8 | | B `aliasband_marg.py` → `aliasband_marg.json` |
| Alias simulation (103 epochs): 0/4000; 96 %; observed Δ 23.4 | | A `alias.py` (rerun, reproduced) |
| 22–40 d band: forward 0/20 000 (p99 9.1, max 20.0); 0.35 % at 12 m/s; 2.8 % at 15; harmonic bands 15.8 vs 20.5, 11.0 vs 22.5, 10.2 vs 23.2; K = 4.5 recovered 92 % | | B `rot2240.py` → `rot2240.json` |
| e < 0.39 (95 %, 103 epochs, emcee, uniform e); 104-epoch eccentric solution e = 0.90, K = 30 m/s with periastron on the discrepant night, ΔlnL = +24 over circular | | B `ecc.py`, `ecc_check.py`, `ecc_check2.py` |
| Chen & Kipping radius 3.4 R⊕ (1800 ppm); Earth-like 2.0 R⊕ (600 ppm, Zeng 2019); TOI-6263.01 130 ppm | | arithmetic from Table 2 (**v12's 2.5 R⊕ was wrong**) |
| Residual rms 4.2 m/s without the discrepant night, 6.3 with it; per-label 1p jitters 7.66/6.23/4.23/3.46 (104) and 3.09/6.34/4.20/3.46 (103) | | inline, `split.py` `fitK` |
| Held-out prediction: D = −5.2 ± 2.7 m/s, noise 4.8 m/s (≈1σ test) | | B `heldout_new.py` logic with approximate times (**v12's ±1.0 omitted the phase error**) |
| Sreenivas et al. 2022: HD 105779 b K 10.42 ± 0.96, ΔlnZ 35.11, rms 3.53, 53 RVs, 2004-02–2019-03; HD 103891 b K 21.12 ± 0.86, ΔlnZ 94.32, rms 3.96, 91 RVs, 2004-02–2018-04 | | their Tables 1–4 (**v12 had K = 5.27 ± 0.45 and 14.78 ± 0.30, wrong**) |
| Bouchy et al. 2009: HD 47186 b P 4.0845, K 9.12 ± 0.18, 22.78 M⊕, e 0.038, 66 RVs, 1583 d, rms 0.91, P_rot 33 d from R'HK, log R'HK −5.01; HD 181433 b P 9.3743, K 2.94 ± 0.23, P_rot 54 d | | their Tables 1–3 |
| FIP (≤2 planets, noise marginalised): 1.70e-4 (104), 1.1e-9 → quoted "< 1e-4" (103, below the method's resolution); ≤1 planet: 0.145 (104), 9.3e-5 (103); TIP1 0.882 / 0.9999; TIP2 0.99986 / 1.0; lnB21 +8.64 / +17.32; narrow-band lnB +8.68 / +18.46; ESS 11–46 | | B `fip.py 240` → `fip_seed11.json` (copy of `fip.json`) |
| Third sinusoid given P0 and 200.9 d: lnB32 −1.15 (104), +5.04 (103); fraction of band mass within 1/T of P0: 0.9998 / 0.99999998 | | B `fip_extra.py` → `fip_extra.json` |
| Fixed-jitter (null ML) frame, for the caveat only: FIP 0.45 / 0.006; Occam 4.70 / 5.33; narrow 5.66 / 11.14 | | B `semian.py`, `semian2.py 20 10` |
| Matérn GP velocity amplitude posterior (uniform 0–20 m/s): median 3.1, 95 % < 7.7 (104); 2.9, < 6.6 (103) | | B `gp_ul.py` → `gp_ul.json` |
| Posterior-typical covariance (5.2 m/s, 1207 d; upper 9.3 m/s, 750 d): Δχ² at P0 = 42.2 / 38.5 (104), 0/3000 exceed | | B `gp_typical.py` → `gp_typical.json` (inputs: medians / 84th pct of `phase3/out/ecc_v1.json` posterior) |
| QP GP, P_rot free 10–100 d: ML velocity amplitude at floor; P = 41.7 d, λ ≈ 345 d, w ≈ 1.4; Δχ² 32.2 / 43.9; 4/5000, 0/5000; posterior A_rv < 5.1 (104), < 3.7 (103) m/s; P_rot 41.5 ± 0.5 d | | B `qpgp.py`, `gp_ul.py` |
| dLW quasi-period: profile peak 41.4 d, +14.0 vs aperiodic, ≥ 8.6 vs other periods (103); 072 only 41.2 d (+8.7); others 36.1 d (+5.8); first half 41.8; second half 36.1 (+3.2); Hα 38.9 d (+5.2); window at 41.4 d = 91st percentile (20–100 d) | | B `qp_profile.py`, `qp_split.py` |
| Evolving-spot forward model (QP GP, P_rot 17–45 d, λ 1–5 rotations, w 0.25–0.8): 0/20 000 at 3.2 m/s rms, 0/10 000 at 6, 39/10 000 at 10 | | B `fwd_qp.py` → `fwd_qp.json` |
| 17–45 d band: bands 15.7 vs 22.2, 13.5 vs 23.2, 16.8 vs 24.3; K = 4.5 recovered 88.5 %; forward 0/20 000, 0.35 % at 12, 2.55 % at 15 | | B `rot1745.py` → `rot1745.json` |
| No-NZP Δχ² 31.9 (104), 44.6 (103); discrepant night 9.5–9.9 × rms (5.5 m/s) of other 103 epochs | | B `nonzp.py` |
| Eccentric (adopted model, Kipping prior): ΔlnZ −2.93, e95 0.32 (104), 0.20 (103), Student-t 0.36 | | P `phase3/out/ecc_v1.json` |
| 200.9-d: +5.98 (Jeffreys 150–260 d) → +3.85 for 10–1000 d (`phase3/p3run.py` P2_WIDE) → +3.46 for 1.05–1000 d (arithmetic) | | P / arithmetic |
| HD 22496 b (Lillo-Box 2021): residual std 0.28 m/s (their Fig. 5 caption); HARPS points from BJD 2452944.8 → 17.4-yr joint baseline; K precision 8.4 % | | their text and Table C.2–C.3 |
| dLW QP null: 40 aperiodic simulations (SE covariance at ML, same epochs), 40-period single-start scan over 20–100 d: max gain 4.56, median 0.76 (observed 14.03) | | B `qp_null.py 3 40` → `qp_null_3.json` |
| Long-lived evolving-spot forward model (λ 5–10 rotations, w 0.8–1.5): 0/10 000 at 3.2 and at 6 m/s rms | | B `fwd_qp_long.py` → `fwd_qp_long.json` |
| Harmonic separations from P0: 38.87 d (9th, 4.319 d) 18 elements; 41.4 d (10th) 47; 36.1 d (8th) 83; exact 9P0 = 38.415 ± 0.025 d | | arithmetic, resolution 1/6536 d⁻¹ |
| Per-season stacked peaks near 40 d: ΔLW 40.6 d, Hα 39.4 d (below 1 % levels) | | P `phase11/prot/season_prot.json`, `Claude outputs/PROT.md` |
| FIP seed check (104, seed 23, 240 draws): lnB1 narrow 8.30, full 3.10; lnB2 11.62; FIP2 2.34e-4; FIP1 0.168; TIP1 0.869; TIP2 0.99980; ESS 22–45 | | B `fip_seed.py 240 23` → `fip_seed23.json` |

## v14 additions

| Quantity | Value | Source |
|---|---|---|
| K at P0 under the QP GP (ML covariance, GLS) | 5.63 ± 0.99 m/s (104), 5.47 ± 0.83 (103) | `analysis/qp_K.py` (uses `qpgp.json`) |
| QP-weighted FAP, all epochs | 4/5000 = 8e-4 | `qpgp.json` (v13 number, now stated as a FAP) |
| Repeat nested-sampling run of the adopted model | lnZ(1p) = −738.17 ± 0.26; P = 4.268371 ± 0.00026 d; K = 5.40 ± 0.79 m/s; T_conj = 6298.35 ± 0.10; A_RV = 2.9 (+2.2/−1.6) m/s; log10 ℓ = 2.84 (+0.23/−0.32) | `gpns.py 1p in 7` → `ns_1p_in_s7.pkl`; `figs_corner.py` → Fig. 4 |
| Ephemeris phase uncertainty (1σ), from that posterior | 0.069 of an orbit (2024 March); 0.088 (end of 2028); corr(P, T_conj) = +0.34 | same pickle |
| Transit limits | TOI-6263.01 transits only for i > 84.69°; b (3.4 R⊕) escapes transit for i < 85.66°; 1/sin(84.69° − 2°) = 1.008 | geometry from Table 3 and a_c = 0.03757 au |
| Second-spectrograph estimate | σK ≈ σ√(2/N): N = 40, σ = 3–4 m/s → K/σK = 8.2–6.2 | arithmetic |
| Chen & Kipping radius at true mass 14.5 M⊕ | ≈ 3.8 R⊕ (3.4 × (14.5/11.8)^0.589) | arithmetic |
| Neptunian ridge | 3.2–5.7 d, mapped for 4–10 R⊕ (overdensity measured at 5.5–8.5 R⊕) | Castro-González et al. 2024, A&A 689, A250 |
| RVBank version | CDS J/A+A/683/A125, table4 corrected 2026-04-20; HD 297396 rows identical (BJD, RVs, errors, dLW, NZP) to GitHub HARPS_RVBank Ver.02 (2024-03-03) | checked by direct comparison |
| ESO programmes | 18: 072.C-0488, 074.C-0364, 078.C-0044, 085.C-0019, 087.C-0831, 089.C-0732, 090.C-0421, 091.C-0034, 092.C-0721, 093.C-0409, 095.C-0551, 096.C-0460, 099.C-0458, 0100.C-0097, 0102.C-0558, 106.21R4.001, 108.222V.001, 183.C-0972 | `ProgID` column of the RVBank rows |


## v15 additions (2026-09-29)

| Quantity | Value | Source |
|---|---|---|
| Held-out spectra (ESO archive, retrieved 2026-09-29) | BJD 2459656.70912, 2460389.65740, 2460629.86180; DRS RV 16388.774, 16374.805, 16386.248 m/s; DVRMS 3.50, 5.71, 3.92 m/s; SN60 57.5, 25.1, 43.9; DRS HARPS_3.8, K5, STAR,SKY; progs 108.222V.001, 106.21R4.001 | headers of `*_ccf_K5_A.fits` in the ancillary tarballs ADP.2022-03-19T01:05:23.556, ADP.2024-03-21T01:00:35.638, ADP.2024-11-16T01:02:11.127; `ancillary/heldout.dat` |
| Frozen criteria on them | C3 pass (no drift), C4 pass, C5 pass (Moon 65°/75°/110°); C1, C2 not applicable | `analysis/heldout_v15.py` |
| Pre-registered contrast | phases 0.804, 0.520, 0.795 from conjunction; D_pred = -4.21 ± 2.87 m/s; D_obs = -12.71 m/s; noise 7.89 m/s; z = -1.01; z0 = -1.61; power P(z0 < −2 \| planet) = 0.083; with photon-noise errors z = −1.27, z0 = −2.10 | `analysis/heldout_v15.py` → `heldout_v15.json` |
| ESO HARPS_RVCAT_V1 vs RVBank RVdrs | 108 spectra matched, max \|diff\| = 0.0005 m/s | `heldout/harps_rvcat_v1_hd297396.csv` (ESO TAP `tap_cat`, 2026-09-29), `heldout_v15.py` |
| Secondary (not pre-registered) absolute test | offset from post-upgrade DRS epochs, 2.2 m/s zero-point allowance: ln B(planet/none) = -0.24 (odds 0.8:1) | `heldout_v15.py` |
| Mount Wilson activity (Gomes da Silva et al. 2021) | S_MW 0.4876; log R'HK median −4.787, weighted mean −4.806 ± 0.004; B−V 1.120 ± 0.003; Teff 4622; [Fe/H] 0.11 | VizieR J/A+A/646/A77 (queried 2026-09-29) |
| Rotation estimate | τc 23.9 d; Ro 1.63; P_rot 39.1 d (31.0–49.2 d at 0.1 dex) Noyes; 38.3 d MH08; v14 input (−4.64, B−V 1.09) gave 28.6 d | `analysis/rotcal.py` |
| Ro(P0) | 0.178 → log R'HK −4.23 (Noyes), −4.31 (MH08) vs observed −4.79 | `rotcal.py` |
| Multiples of P0 in 30–50 d | 8th–11th; P0/P_rot 0.085–0.142; 9P0 = 38.415 d | `rotcal.py` |
| Period stability | 2.72e-5 → 1.5 min for a 39-d period | `rotcal.py` |
| 30–50 d residual bands | 30–50: 13.9 vs 20.2; 15–25: 13.7 vs 21.3; 10–16.67: 11.0 vs 22.0; K = 4.5 recovered 0.895 | `analysis/rot3050.py` → `rot3050.json` |
| 30–50 d coherent forward model | 4.5 m/s: 0/20 000 (p99 9.3, max 18.0); 12 m/s 0.27 %; 15 m/s 1.95 % | `rot3050.py` |
| Evolving-spot forward models, 30–50 d | short-lived (1–5 rot., w 0.25–0.8): 0/10 000 at 3.2 and 6 m/s, 0.30 % at 10; long-lived (5–10 rot., w 0.8–1.5): 0/10 000 at 3.2 and 6, 0.26 % at 10 | `analysis/fwd_qp_3050.py` → `fwd_qp_3050.json` |
| Ninth-harmonic worst case | P_rot within 1 % of 38.415 d, w 0.15–0.4, 1–10 rotations: 0/10 000 at 3.2 and 6 m/s, 0.36 % at 10 | `fwd_qp_3050.py` |
| Simulation counts in the abstract | 150 000 = coherent 20 000 (25–40 d, v12) + 20 000 (30–50 d, v15) + QP 50 000 (v13) + 60 000 (v15), all at amplitudes ≤ 6 m/s rms or the 4.5-m/s 90 %-recovery bound | arithmetic |
| TOI-6263.01 (ExoFOP, 2026-09-29) | depth 193 ± 17 ppm, S/N 9, QLP s62 FFI, sectors 8, 9, 35, 36, 62; TESS and TFOP disposition PC; note "low SNR; slight depth-aperture correlation"; TFOP SG1 LCO-CTIO 1 m 2023-04-25, ip, 137/139 neighbours cleared to 2.5′, non-detection on target (predicted 0.19 ppt) | ExoFOP TIC 293547742 JSON; file 914139 (R. Schwarz notes) |
| Zorro sensitivity (ExoFOP files 1029891, 1029890) | 832 nm: Δm 3.83 at 0.10″, 7.23 at 0.525″, 8.29 at 1.175″ (outer edge); ExoFOP summary Δm 7.02 at 0.5″ (832) and 5.63 at 0.5″ (562); PI Howell | ExoFOP |
| Yu et al. (2024) | HD 297396 in their 268-star sample (Table 3: Teff 4622, log R'HK −4.806, data 2004-01 to 2015-05); not among the 49 vetted rotation periods | arXiv:2401.05528 |

### v15 additions after the referee check (REFEREE_v15.md)

| Quantity | Value | Source |
|---|---|---|
| Held-out power | registered 0.083; anticipated 0.212 (v13 numbers: D −5.2 ± 2.7, noise 4.8) and 0.169 (header BJDs, 3.9 m/s only) → "about 20 %" | `analysis/heldout_v15.py` |
| Held-out, photon-noise errors (not pre-registered) | CCF NOISE 1.34, 3.56, 1.62 m/s; noise 6.05; z = -1.27; z0 = -2.10 | `heldout_v15.py` |
| DRS/SERVAL error ratio | STAR,SKY 2021 RVBank spectra: e_RVdrs/e_DRVmlc = 2.52, 3.37; STAR,DARK post-upgrade median 1.15 | inline check on `HD297396_rvbank_full.csv` |
| Nightly zero-point rms | 0.98 m/s (std of NZPdrs over post-upgrade nights) | `heldout_v15.py` |
| Secondary test, corrected covariance (nightly NZP on diagonal, 2 m/s shared) | ln B = -0.29 (odds 0.75:1) | `heldout_v15.py` |
| 2024 March spectrum vs the other two | −12.9 m/s below the null offset; 2022 and 2024 Nov differ by 2.5 m/s | `heldout_v15.py` printout |
| Harmonics of 9P0 in the planet-fit residuals (104) | fundamental K = 0.46 ± 0.79; max K95 over k = 1–8, 10, 11 = 3.00 m/s; max Δχ² = 3.8; K at P0 = 5.72 | `analysis/comb9.py` → `comb9.json` |
| Same, 103 epochs | max K95 = 2.33; max Δχ² = 5.3 | `comb9.py` |
| QP-kernel variance in the 9th harmonic | w = 0.15: 0.0065 (K9 = 0.69 m/s at 6 m/s rms; 48 m/s rms needed for K9 = 5.5); w = 0.2: 7.7e-04; w = 0.4: 1.3e-07 | `comb9.py` (2 I_n(κ) e^−κ, κ = 1/4w²) |
| 170 rotations | 6536 d / 38.415 d = 170 | arithmetic |
| TOI-6263.01 | R_p = 1.12 (+0.06/−0.06) R⊕ (193 ± 17 ppm, R★ 0.739 ± 0.026); catalogue 0.93 R⊕ implies R★ = 0.614 R☉; M_p (Chen & Kipping terran) 1.5 M⊕; K = 0.78 (+0.37/−0.25) m/s; Δ = 8.9 (+0.5/−0.4) mutual Hill radii; all 2×10⁵ draws above 2√3 | `analysis/toi.py` → `toi.json` |
| Coherence in rotations | τ > 2300 d = 46 rotations at 50 d | arithmetic |
| Coherent forward model shown in Fig. 6 (30–50 d, seed 3051) | 0/20 000 at 4.5 m/s, p99 9.0, max 20.1; 12 m/s 0.18 %; 15 m/s 2.15 %; Keplerian 0.72 | `analysis/fwd2_3050.py` → `fwd_3050.npz`; figure `figs_fwd3050.py` (the independent `rot3050.py` run gave p99 9.3, max 18.0, 0.28 %, 1.95 %) |
| Photometric plan, P_rot ~ U(30,50) | 2 mmag/night: 25 nights over 60 d 0.95, over 90 d 0.99 at 5 mmag; 3 mmag/night: 25/60 0.63, 30/90 0.88, 40/120 0.98 | `analysis/phot_plan.py` → `phot_plan.json` |
| Mount Wilson log R'HK of comparison stars (GdS21 medians) | HD 105779 −4.887, HD 103891 −5.123, HD 47186 −5.070, HD 181433 −5.128; HD 22496 not in catalogue | VizieR J/A+A/646/A77 (2026-09-29) |
| Seasonal stacking band and injection periods | 10–60 d; injections at 25 and 30 d | `Claude outputs/PROT.md` (phase 11) |
| Perdelwitz et al. (2024) scale | their Fig. 8 and Sect. on comparison with Gomes da Silva et al.: "Cooler stars appear more active ... relative to the values given by Gomes da Silva et al." | A&A 683, A125 full text |
