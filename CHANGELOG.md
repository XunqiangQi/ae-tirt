# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-02-15

### Added
- Initial academic package scaffold for AE-TIRT.
- Modular architecture: `models`, `data`, `training`, `evaluation`, `experiments`, `utils`, and `config`.
- AE-TIRT core model and TIRT simulator extraction from legacy scripts.
- Batch experiment engine with Study-1 and Study-2 condition generators.
- Result writers for trait and item-parameter recovery outputs.
- Reproducible examples and script entry points.

### Changed
- Standardized project structure for research reproducibility.
- Promoted paper-fixed experiment settings in `scripts/run_all_experiments.py`.
- Expanded documentation for methodological and applied workflows.

### Notes
- This release focuses on computational workflows and parameter recovery.
- Formal uncertainty quantification (SEs/CIs) remains out of scope in this version.
