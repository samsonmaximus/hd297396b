# The search that found the signal (6 September 2026)

These are the stored outputs of the survey described in Sect. 2.2 of the paper. The survey ran
on 6 September 2026 on HARPS-RVBank ver02 (the public GitHub release), before any of the analysis
in the paper.

| File | What it is |
|---|---|
| `survey.py` | The survey code as it ran. Generalised Lomb–Scargle with pre/post-2015 offsets and a linear trend fitted at every frequency, from 1 d to the baseline, three peaks per star by prewhitening. Stars with ≥ 20 velocities over > 100 d, kept if they had ≥ 15 nights after binning. |
| `vet.py` | The vetting functions as they ran: catalogue cross-match (by coordinates and by name, with aliases and harmonics), indicator periodograms and correlations, the pre/post-upgrade split. |
| `rvbank_gls_survey_all_stars.csv` | Survey output: one row per searched star (1328), three peaks each. |
| `rvbank_vetting_results.csv` | Vetting output for the 109 planet-like peaks on 98 stars. |
| `HD297396_report_2026-09-06.html` | The notes written on the day of the search, including the triage table that Sect. 2.2 quotes. |
| `search_counts.py`, `search_counts.log` | Regenerate the counts below and the archive-wide trials arithmetic of Sect. 4.2. |

## Where each number in Sect. 2.2 comes from

| Number | Source |
|---|---|
| 1328 searched stars | Rows of `rvbank_gls_survey_all_stars.csv` (`search_counts.py`). |
| 227 peaks matching no catalogued planet | The triage table of the 6 Sept report. The catalogue snapshots it used were not stored, so this count cannot be regenerated. The table does not add up on its own: 659 peaks, 268 matched, 227 unmatched leaves 164 peaks unaccounted for. Probably they were removed at the periods of the observing window (the year, the day, the lunar month and their fractions; `vet.py` has a function for it), since none of the 109 planet-like peaks lies at such a period while 140 of the 746 stored peaks with FAP < 10⁻⁴ do. The paper therefore says "in the triage recorded that day". |
| 109 planet-like peaks on 98 stars | Rows of `rvbank_vetting_results.csv`. `search_counts.py` checks the four cuts (P < half the baseline, M sin i < 13 M_Jup, K > 2.5 times the median error, ≥ 25 nights) on every row. |
| 48 after the activity-indicator screen | The triage table of the report. The screen's code was not stored. Requiring that no indicator has a false-alarm probability below 1 % at the candidate period, and that no indicator correlates with the velocities at \|r\| ≥ 0.4, gives exactly 48 peaks on 44 stars, HD 297396 among them. This is a reconstruction. It is probably the rule that was used, but the record cannot prove it. |
| The final choice | Manual review of the 48 for rotation-like periods, consistency across the 2015 fibre upgrade, and sampling. The report names these criteria. No written rule was followed. |

One count in the report is not used in the paper: it gives 659 peaks with FAP < 10⁻⁴, while the
stored survey table has 746. The difference was probably a filter applied before counting, but it
was not recorded.

## Archive-wide trials factor (Sect. 4.2)

The survey searched 1328 stars from 1.0 d to the baseline. The paper's false-alarm band is
1.05–1000 d, which is 1.050 times narrower in frequency (periods beyond 1000 d add less than 0.1 %).
The archive-wide numbers are FAP × 1328 × 1.05: 1.95 expected false alarms of this height with
all 104 epochs and 9.3 × 10⁻³ with 103.

This scaling is first-order (Baluev 2008). Rerunning the 10 000 noise draws over 1.0–1000 d would
not make it more precise. The 104-epoch FAP rests on 14 exceedances, a ±27 % Poisson uncertainty,
far larger than the 5 % band correction.

## About the report

The report is released unedited, except that a first name was replaced by "[the author]" in three
places. It was written as private notes on the day of the search, before any of the tests in the
paper. Its significance figures (an analytic FAP of 1.9 × 10⁻¹⁰ with fixed errors and no jitter,
"7–8σ") and its 85–90 % probability estimate are superseded by the paper.
