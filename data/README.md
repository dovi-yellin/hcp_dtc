# Data Directory

This directory is reserved for local datasets used by the HCP DTC analyses.

Large raw, intermediate, or derived datasets should not be committed to Git. Keep them on local storage, institutional storage, or another controlled data system, and document the expected filenames and provenance here as the analysis pipeline stabilizes.

`figure_S3/` contains the small, intentionally tracked **simulation-only** source summaries for the revised Supplementary Figure S3:

- `tau120_network_means.csv`: 210 rows, one four-window fALFF mean for each of 14 network-and-input realizations at 15 actual stability margins.
- `provenance.json`: model/observation settings, checksums, sampling definition and the verified finite-range fit.

These files contain no HCP observations, participant identifiers or participant-derived statistics. They are sufficient for `python scripts/make_figure_S3.py` and are narrowly allowlisted in `.gitignore`; other local datasets remain ignored.
