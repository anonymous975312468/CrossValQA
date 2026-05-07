# Changelog

All notable changes to MutQA and the CrossValQA pipeline will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The dataset version tracks the canonical MutQA release; the pipeline code follows the same versioning. Errata identified post-release are listed here with affected record IDs (where applicable) and remediation status.

## [1.0.0] — 2026-XX-XX

### Initial release

- First public release of MutQA accompanying the NeurIPS 2026 Evaluations & Datasets Track submission.
- ~401,000 question-answer pairs total; ~330,000 cross-grounded under the CrossValQA bidirectional validation protocol.
- Homology-aware 80/20 train/test split (UniProt-grouped, MMseqs2 + GraphPart-style audit; max cross-split identity 39.7%).
- CrossValQA pipeline source code released under MIT.
- Dataset released under CC-BY-4.0.
- Croissant 1.0 metadata with Responsible AI extension fields included.
- Gebru-format datasheet included as paper appendix and as `DATASHEET.md` in this repository.
