## License

- **Dataset (MutQA):** [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/). Use, redistribution, modification, and commercial use are permitted with attribution.
- **Pipeline code (CrossValQA):** [MIT License](LICENSE).

Source PubMed content remains subject to the licenses and terms of its respective publishers. MutQA references source articles by PMID and does not redistribute full-text source content.

## Citation

If you use MutQA or the CrossValQA pipeline in your research, please cite:

```bibtex
@inproceedings{mutqa2026,
  title     = {MutQA: A Citation-Grounded Question Answering Benchmark for Mutation-Aware Text Models},
  author    = {ANONYMIZED},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```

## Datasheet

Full Gebru-format datasheet is available at [`DATASHEET.md`](DATASHEET.md). It documents motivation, composition, collection process, preprocessing, intended uses, distribution, and maintenance.

## Croissant Metadata

The dataset is documented with a [Croissant 1.0](http://mlcommons.org/croissant/) metadata file including [Responsible AI (RAI) extension](http://mlcommons.org/croissant/RAI/) fields. The Croissant file is available at [`croissant.json`](croissant.json) and on the [HuggingFace dataset page](https://huggingface.co/datasets/<HF_ORG>/MutQA).

## Maintenance

MutQA is maintained by the originating research lab. The v1.0.0 release described in the accompanying paper is the canonical reproduction reference and will remain available indefinitely on HuggingFace. Errata, version changes, and breaking changes will be documented in [`CHANGELOG.md`](CHANGELOG.md). Superseded versions are retained for reproducibility.

**Reporting issues.** Please file errors, requested corrections, or feature requests via [GitHub Issues](https://github.com/<ORG>/<REPO>/issues). We commit to responding to legitimate issues within 30 days.

**Contributions.** Pull requests are welcome for pipeline improvements. For dataset extensions (including applications of CrossValQA to new domains), please open an issue first to discuss scope.

## Author Responsibility

The authors confirm that, to the best of our knowledge, MutQA does not violate the rights of original PubMed content owners. Derived QA pairs are released as research-use commentary under CC-BY-4.0 with full source attribution preserved via PMID. The authors bear responsibility for the dataset and any rights violations identified post-release; please report concerns via GitHub Issues.

## Intended Use and Out-of-Scope Use

MutQA is intended for **research evaluation** of mutation-aware text models. See [`DATASHEET.md`](DATASHEET.md) for a full discussion of intended use cases (UC1-UC5) and limitations. The dataset **must not be used** for clinical outcome prediction, variant interpretation in patient care, drug response prediction, or any application that conflates literature-grounded model performance with biological or clinical correctness.
