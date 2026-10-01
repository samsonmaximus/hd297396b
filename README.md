# HD 297396 b — analysis code and data

Code, data and pre-analysis documents behind

> **Fraser, S. (2026). A 4.27-day Neptune-mass planet candidate around the K dwarf HD 297396 in
> 18 years of archival HARPS velocities.** *Submitted to the Open Journal of Astrophysics.*
> arXiv: ARXIVID

The paper reports a 4.27-d signal in 104 nightly HARPS velocities of the K3–K4 dwarf HD 297396
(TOI-6263), spanning 17.9 years. It is tested against false alarms, rotation, instrument effects
and the archive itself:

| Quantity | Value |
|---|---|
| Period | 4.26837 ± 0.00027 d |
| Semi-amplitude | 5.5 ± 0.8 m/s |
| Minimum mass | 11.8 ± 1.9 M⊕ |
| False-alarm probability (all epochs) | 1.4 × 10⁻³ (1.7 expected false alarms across the archive) |
| Phase coherence | within 15° over 1531 orbits |
| Status | **Planet candidate.** All velocities are from one instrument, and the all-epoch significance depends on one discrepant night. One season with a second spectrograph will decide it. |

## Layout

| Folder / file | Contents |
|---|---|
| `analysis/` | Reproduction scripts and their outputs (v12–v15). `analysis/README.md` maps each script to the paper section it produces. |
| `phase3/` | The model-ladder pipeline that produces Table 3 (GP noise models, nested sampling, eccentricity, injection–recovery). `phase3/PREREGISTRATION_index.md` holds the pre-analysis criteria. |
| `data/HD297396_rvbank_full.csv` | The 108 HARPS-RVBank rows for HD 297396 (before night binning and rejection). |
| `ancillary/` | Machine-readable nightly velocities and indicators (104 epochs) and the three later HARPS spectra, CDS format. Same files as the arXiv ancillary upload. |
| `heldout/` | ESO HARPS RV-catalogue rows used for the held-out test (Sect. 5.2). |
| `NUMBERS.md` | Provenance: the script and output file behind every number in the paper. |

## Reproduce

Python ≥ 3.10 with `numpy scipy pandas matplotlib`. The GP, sampling and figure scripts also use
`dynesty`, `emcee`, `george`, `astropy` and `corner` (`pip install -r requirements.txt`).

```bash
cd analysis
python core.py        # rejection criteria and nightly binning -> 104 epochs
python gpdata.py      # writes frozen104.csv, read by the GP scripts
python pg.py          # periodogram, per-label offsets and jitters
```

Two scripts (`counts.py`, `xstar.py`) need the full HARPS-RVBank table, which is not stored here:
download `table4.dat.gz` from CDS catalogue
[J/A+A/683/A125](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/683/A125) into `analysis/data/`.

`analysis/heldout_new.py` is the decision rule for the held-out test, frozen on 2026-09-25
before the three later spectra were retrieved. It is kept byte-for-byte unedited, so its
comments use the section numbers of the version it was written against.

## Pre-registration: what it does and does not prove

The pre-analysis criteria were written before the runs they govern and are quoted verbatim, with
their dates, in `phase3/PREREGISTRATION_index.md`. This deposit was made after those runs. No
independent timestamp predates them, so this repository records the criteria; it cannot prove
when they were written.

## Use of AI tools

The analysis code was written with the assistance of a large language model (Claude, Anthropic),
under the author's direction. Every decision about data, models and claims was the author's. How
the code and results were checked is described in the paper's acknowledgements.

## Citation and licence

Cite the paper above and this archive: Zenodo DOI ZENODODOI (see `CITATION.cff`).
Code: MIT (`LICENSE`). Data, results and documentation: CC BY 4.0 (`LICENSE-DATA.md`). The
velocities derive from HARPS-RVBank (Perdelwitz et al. 2024, A&A 683, A125); please cite it too.
