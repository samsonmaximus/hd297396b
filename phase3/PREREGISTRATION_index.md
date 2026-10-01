# Pre-analysis documents released with the paper

Three criteria sets were fixed in writing before the runs they govern. Two of
them were recorded inside the phase reports that pre-date those runs rather than
as standalone files; they are reproduced here verbatim, with their source and
date, so that a reader can check that nothing was written after the fact. The
third is a standalone document.

**Nothing in this file was composed after the runs it governs.** Where a
criterion exists only as a passage inside an earlier report, that is stated and
the passage is quoted rather than paraphrased. The only change to any quotation is one
redaction: a personal first name is replaced by "[the author]".

---

## (i) The data freeze and the outlier policy

**Source:** `phase1/phase1_report.md` §1, dated 2026-09-08, written before any
correlated-noise run. Quoted:

> **Pre-registered rejection criteria.** Fixed on instrument/observing grounds
> before looking at which nights they hit. None is residual-based, so none is
> circular with respect to the Keplerian being fitted.
>
> | | criterion | rejects |
> |---|---|---|
> | C1 | `Flag != 0` (RVBank: 0 = good) | 0 |
> | C2 | \|d_i − med(d)\| > 5·√(e_DRS² + e_SERVAL²), d = RVdrsnzp − DRVmlcnzp, **median taken per fibre era** | 1 |
> | C3 | \|DRIFT\| > 3.0 m/s (simultaneous-reference failure) | 0 |
> | C4 | `SNRDRS` < 20 (order 50) | 1 |
> | C5 | Moon separation < 30° **and** illumination > 50% | 0 |

and, on the epoch that passes all five:

> The **+51 m/s night** (BJD 2454922.5270) trips **none** of the five criteria
> and is kept. [...] That is the honest statement for the paper: the detection
> does not depend on the night, but the *strength* of the detection does.

The accompanying commitment, also in §4 of that report, is that every subsequent
result is reported both with and without that epoch. It is.

---

## (ii) The second-signal rule

**Source:** `phase4/phase4_report.md` §5, dated 2026-09-08, written before any
correlated-noise run. Quoted:

> Under the pre-registered rule — *"if 2p1c2c wins by > 6 and 200.5 d isn't a
> window peak, it's a second candidate signal"* — the 200.9 d signal **does not
> qualify**. It fails the window leg outright [...]. The rule therefore **does
> not grant "second candidate signal" status**, and it should not be quietly
> amended now that the outcome is known — a pre-registered criterion relaxed
> after seeing the result is worth less than no criterion at all.

The same section states, in advance, the form any amendment would have to take:

> If [the author] wants to promote it, the amendment has to be pre-registered *before*
> the next data set and stated as an amendment, e.g. "a signal coincident with a
> window peak may be promoted if the alias-parent test, the trend-removal test
> and the indicator test all pass and two independent pipelines agree on
> ΔlnZ > 6 in the same data set."

The correlated-noise runs moved the evidence leg (both pipelines now pass on the
primary data set) and left the window leg where it was. The rule requires both.
It has not been relaxed. The contemplated amendment is reported in the paper as
something a *future* analysis could adopt, not as something adopted here.

---

## (iii) The seeing-term protocol

**Source:** `phase3/PREREGISTRATION_step2.md`, a standalone document written
2026-09-08 before the first seeing-term run was executed. It fixes the three
functional forms and their priors, the treatment of the epochs with no DIMM
record, the rule that the form is chosen on the planet-free evidence alone, four
numerical read-out criteria, and the commitment that seeing would under no
outcome be converted into a rejection criterion.

Two departures are recorded in the paper rather than smoothed over:

1. **The selection rule conflicts with the document's own designation.** The
   rule ("choose on the planet-free evidence") selects the 1.5″ step form, by
   1.09 nats. The document separately designates the threshold-free power law as
   PRIMARY. We followed the designation, which takes the lower of the two
   evidences.
2. **The choice of kernel was never pre-registered.** Matérn over quasi-periodic
   is a judgement, made on a 1.63-nat margin, in the direction that lowers the
   evidence for the planet.

---

## What was *not* pre-registered

Everything else, and in particular: the choice of Matérn over quasi-periodic
kernel; the decision to run a white-noise control through the same code; the
leave-one-out control on the seeing term; the decision to quote the adopted
evidence as the mean of repeated runs; and the decision to compute detection
limits in the whitened frame. All of these were added during the analysis
because they seemed necessary, and all of them are reported whichever way they
came out.
