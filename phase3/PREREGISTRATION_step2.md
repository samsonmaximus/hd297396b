# Pre-registration — the seeing-dependent noise term (work order, step 2)

Written 2026-09-08, **before any seeing-term run was executed**. Fixed at the
time of writing: the functional forms, the priors, the treatment of the nights
without a DIMM value, the rule for choosing among the forms, and the read-out.

## Predictor

Nightly-mean DIMM seeing from the ESO archive headers of the HARPS exposures,
joined onto the frozen `data-v1` rows by the same night key already used for
dLW. This is instrumental metadata. Phase 0 established from it — *not* from any
residual of any fitted model — that the +51 m/s night (BJD 2454922.527) was
taken at 1.97", the second-worst of 111 exposures of this target against a
median of 0.90". On `data-v1`, 83 of 104 nights carry a value and 5 exceed 1.5".

## Forms

All three add a term to the RV variance only, and all three give the 21 nights
without a DIMM value their own free jitter `s_nodimm ~ U(0, 15)` m/s rather than
an imputed seeing, because imputing the median would quietly assert those nights
were good.

- **A — power (PRIMARY).** `sigma_i^2 += [gamma * (s_i / 0.90")^q]^2`,
  `gamma ~ U(0, 15)` m/s, `q ~ U(0, 10)`. Threshold-free and monotone: the
  steepness is fitted rather than chosen. The 0.90" normalisation is the sample
  median seeing, a property of the observing record, not of any residual.
- **B — hinge.** `sigma_i^2 += [gamma * max(0, s_i - 1.00")]^2`,
  `gamma ~ U(0, 40)` m/s/". The 1.00" break is the HARPS fibre aperture, an
  instrumental constant known independently of this data set: seeing in excess
  of the fibre aperture is what produces variable decentring.
- **C — threshold (DECLARED POST HOC).** `sigma_i^2 += s_bad^2 * 1[s_i > 1.50"]`,
  `s_bad ~ U(0, 40)` m/s. The 1.50" break was read off a Phase-0 *residual*
  table, so this form is post hoc by construction and is reported as such, with
  sensitivity at 1.30" and 1.70".

**The symmetric form `gamma^2 (s_i - s_bar)^2` given in the work order is not
run.** It asserts that better-than-median seeing degrades the RV, which
contradicts the decentring mechanism; A supersedes it as the threshold-free
primary.

## Selection rule

The choice among A, B and C is made on `lnZ` of the **planet-free (0p)** model
on `serval_in` alone, fixed here before any 1p1c run with a seeing term is
inspected. Choosing the noise model on the planet-free evidence is what stops
the choice from being driven by what it does to the planet.

## Read-out

Evaluated on `serval_in`, Gaussian likelihood, Matern-3/2 GP. Baselines already
in hand: `dlnZ(1p1c - 0p)` = +8.87 (Gaussian GP), +11.72 (GP + Student-t),
+27.44 (Gaussian GP, +51 night deleted); `sigma_K/K` = 15.1% in / 6.6% out;
`s_rv(HARPS_pre_072)` = 7.77 m/s in / 0.96 m/s out.

1. **The term is real** iff `lnZ(0p + seeing) - lnZ(0p) > 3`.
2. **It matches the robust likelihood** iff `dlnZ(1p1c - 0p) >= 11.7`.
3. **It delivers the precision** iff `sigma_K/K <= 10%`.
4. **It explains the night** iff the `HARPS_pre_072` jitter median is < 5 m/s.

Outcomes, fixed in advance:

- **1–4 all hold** → the "third door": the paper keeps every night, keeps the
  freeze, adopts `+51 in` with a seeing term as the headline, and quotes a mass
  to <= 10%.
- **1–2 hold, 3–4 do not** → the seeing term is reported as a modelling
  improvement; the headline stays the Phase-1 freeze row without it.
- **1 fails** → negative result: the +51 excursion is not a seeing artefact at
  the level these RVs can show, and that is stated as the finding.

Under no outcome is seeing converted into a *rejection* rule for this paper.
That would be a post-hoc amendment to a pre-registered freeze, which is exactly
what Phase 4 section 5 warns against once the outcome is known.
