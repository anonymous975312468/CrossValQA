# M2TQA Grounded Dataset Statistics

## Overview

This document summarizes the final grounded M2TQA dataset after gene anchoring, NCBI Gene ID recovery, and UniProt mapping.

---

## Dataset Summary

| Metric | Count |
|--------|------:|
| **Total Grounded Records** | 416,188 |
| **Unique NCBI Gene IDs** | 4,689 |
| **Unique UniProt Accessions** | 4,685 |
| **Unique (Gene, Mutation) Pairs** | 126,144 |
| **Unique Mutation Strings** | 91,329 |

---

## Grounding Methods

| Method | Count | Percent |
|--------|------:|--------:|
| Direct (PubTator annotation) | 290,959 | 69.9% |
| Recovered via Gene ID (taxon inferred) | 76,295 | 18.3% |
| Recovered via Symbol Lookup (NCBI gene info) | 48,934 | 11.8% |
| **Total** | **416,188** | **100%** |

### Symbol Recovery Details

- Attempted: 55,262 orphaned records
- Symbol found in NCBI: 49,069 (88.8%)
- UniProt mapping found: 48,934 (88.5%)

---

## QA Pair Grounding Status

| Status | Count | Percent |
|--------|------:|--------:|
| cross-grounded | 343,396 | 82.5% |
| ungrounded | 58,310 | 14.0% |
| grounded | 14,482 | 3.5% |
| **Total** | **416,188** | **100%** |

---

## Question Type Coverage

| Question Type | Count | Percent |
|---------------|------:|--------:|
| -function | 103,048 | 24.8% |
| -pathogenicity | 84,759 | 20.4% |
| -disease | 79,827 | 19.2% |
| **stratify further**other | 51,688 | 12.4% |
| frequency | 33,160 | 8.0% |
| clinical | 32,132 | 7.7% |
| mechanism | 14,230 | 3.4% |
| inheritance | 13,452 | 3.2% |
| (other combined) | 3,892 | 0.9% |
| **Total** | **416,188** | **100%** |

---

## Mutation Type Coverage

| Mutation Type | Count | Percent |
|---------------|------:|--------:|
| missense | 146,054 | 35.1% |
| other | 93,203 | 22.4% |
| substitution (coding) | 82,909 | 19.9% |
| deletion | 33,558 | 8.1% |
| nonsense/stop | 15,198 | 3.7% |
| splice site | 10,889 | 2.6% |
| frameshift | 9,665 | 2.3% |
| duplication | 8,264 | 2.0% |
| coding variant (other) | 6,975 | 1.7% |
| protein variant (other) | 4,215 | 1.0% |
| insertion | 3,760 | 0.9% |
| dbSNP rsID | 1,498 | 0.4% |
| **Total** | **416,188** | **100%** |

---

## Species Distribution

| Taxon ID | Species | Count | Percent |
|----------|---------|------:|--------:|
| 9606 | Human | 412,224 | 99.0% |
| 10090 | Mouse | 2,982 | 0.7% |
| 7955 | Zebrafish | 423 | 0.1% |
| 7227 | Fruit fly | 198 | 0.0% |
| 10116 | Rat | 177 | 0.0% |
| 4932 | Yeast (S. cerevisiae) | 158 | 0.0% |
| 9031 | Chicken | 15 | 0.0% |
| 6239 | C. elegans | 11 | 0.0% |
| **Total** | **8 species** | **416,188** | **100%** |

---

## Top 10 Genes by QA Pair Count

| Rank | Gene | UniProt | QA Pairs | % of Total |
|:----:|------|---------|--------:|:----------:|
| 1 | BRCA1 | P38398 | 12,861 | 3.1% |
| 2 | GJB2 | P29033 | 5,830 | 1.4% |
| 3 | BRCA2 | P51587 | 4,545 | 1.1% |
| 4 | LDLR | P01130 | 3,959 | 1.0% |
| 5 | CFTR | P13569 | 3,847 | 0.9% |
| 6 | MLH1 | P40692 | 3,767 | 0.9% |
| 7 | ABCA4 | P78363 | 3,683 | 0.9% |
| 8 | ATP7B | P35670 | 2,815 | 0.7% |
| 9 | TP53 | P04637 | 2,633 | 0.6% |
| 10 | ATM | Q13315 | 2,453 | 0.6% |
| | **Top 10 Total** | | **46,393** | **11.1%** |

---

## Coverage Analysis

### Original M2TQA Dataset
- Total QA records: 473,243

### Gene Anchoring Results
- Successfully anchored: 414,458 (87.6%)
- Missing gene anchor: 58,785 (12.4%)

### Final UniProt Grounding
- Grounded to UniProt: 416,188 (87.9%)
- Not grounded: 57,055 (12.1%)

### Why Records Were Not Grounded

| Reason | Count |
|--------|------:|
| Unsupported species (Xenopus, bacteria, etc.) | 38,090 |
| Gene not in UniProt mapping | 9,114 |
| Symbol not found in NCBI | 6,193 |
| No gene info in record | 3,523 |
| Symbol found but no UniProt | 135 |

---

## Files

| File | Description |
|------|-------------|
| `m2tqa_grounded_final.jsonl` | Final grounded dataset (416,188 records) |
| `grounding_stats_final.json` | Detailed grounding statistics |
| `m2tqa_grounded_statistics.md` | This summary document |

---

## Notes

1. **Gene Family Merging**: 4 gene families (CALM1/2/3, SMN1/2, HBA1/2) encode identical proteins, resulting in 4,689 Gene IDs mapping to 4,685 UniProt accessions.

2. **Cross-Grounded Quality**: 82.5% of records are cross-grounded, meaning both Phase 1 and Phase 2 rationales mutually support the answers.

3. **Clinical Relevance**: Top genes are clinically significant disease genes including hereditary cancer (BRCA1/2, MLH1, TP53, ATM), hearing loss (GJB2), cardiovascular (LDLR), cystic fibrosis (CFTR), and retinal disease (ABCA4).

---

*Generated: 2026-01-26*

---

## MutaDescribe Comparison

### Dataset Overview

| Dataset | Records | UniProt IDs | (UniProt, Mutation) Pairs |
|---------|--------:|------------:|--------------------------:|
| MutaDescribe | 171,147 | 20,946 | 171,147 |
| M2TQA (UniProt-mapped) | 416,188 | 4,685 | 126,142 |
| M2TQA (missense only) | 153,741 | - | 47,328 |

### UniProt ID Overlap

| Metric | Count | Percent |
|--------|------:|--------:|
| MutaDescribe UniProt IDs | 20,946 | 100% |
| M2TQA UniProt IDs | 4,685 | 100% |
| **Overlap (in both)** | **3,536** | - |
| % of MutaDescribe | - | 16.9% |
| % of M2TQA | - | 75.5% |
| MutaDescribe only | 17,410 | - |
| M2TQA only | 1,149 | - |

### (UniProt, Mutation) Pair Overlap

| Metric | Count | Percent |
|--------|------:|--------:|
| MutaDescribe pairs | 171,147 | 100% |
| M2TQA pairs (missense, normalized) | 47,328 | 100% |
| **Overlap (in both)** | **6,750** | - |
| % of MutaDescribe | - | 3.9% |
| % of M2TQA (missense) | - | 14.3% |
| MutaDescribe only | 164,397 | - |
| M2TQA only | 40,578 | - |

### Mutation Type Compatibility

MutaDescribe only contains **missense mutations** (e.g., F294A format).

| M2TQA Mutation Type | Count | MutaDescribe Compatible? |
|---------------------|------:|:------------------------:|
| coding DNA (c.) | 100,724 | No |
| other | 97,479 | No |
| missense (3-letter) | 81,550 | Yes |
| missense (1-letter) | 72,191 | Yes |
| deletion | 33,552 | No |
| frameshift | 9,628 | No |
| duplication | 8,264 | No |
| nonsense | 7,542 | No |
| insertion | 3,760 | No |
| dbSNP rsID | 1,498 | No |

**Only 36.9% of M2TQA mutations (153,741) can be compared with MutaDescribe.**

### Key Findings

1. **UniProt Coverage**: 75.5% of M2TQA's UniProt IDs are in MutaDescribe, but M2TQA only covers 16.9% of MutaDescribe's proteins.

2. **Variant Overlap**: Only 6,750 (UniProt, mutation) pairs overlap - the datasets are largely **complementary**.

3. **Mutation Diversity**: M2TQA covers diverse mutation types (deletions, insertions, frameshifts, splice variants) that MutaDescribe does not include.

4. **M2TQA Unique Contribution**: 40,578 missense variants in M2TQA are not in MutaDescribe.

### Gene-Level Delta

**1,149 UniProt IDs are unique to M2TQA** (not in MutaDescribe).

Top 10 M2TQA-only genes by variant count:

| UniProt | Gene | Variants |
|---------|------|--------:|
| Q9UKN7 | MYO15A | 1,412 |
| P35475 | IDUA | 1,044 |
| Q8N726 | CDKN2A | 931 |
| P34059 | GALNS | 811 |
| Q96LT7 | C9orf72 | 710 |
| P08519 | LPA | 470 |
| Q8TF17 | SH3TC2 | 431 |
| Q8TCU4 | ALMS1 | 427 |
| P20929 | NEB | 390 |
| P48754 | Brca1 | 381 |

### Depth Analysis: Variants per Gene

| Variants/Gene | M2TQA Genes | MutaDescribe Genes |
|---------------|------------:|-------------------:|
| 1 | 56 | 2,522 |
| 2-5 | 736 | 9,534 |
| 6-10 | 793 | 4,310 |
| 11-25 | 931 | 3,380 |
| 26-50 | 720 | 926 |
| 51-100 | 596 | 222 |
| 101-500 | 701 | 52 |
| **500+** | **152** | **0** |

### Depth Summary Statistics

| Metric | M2TQA | MutaDescribe |
|--------|------:|-------------:|
| Total genes | 4,685 | 20,946 |
| Total variants | 416,188 | 171,147 |
| **Mean variants/gene** | **88.8** | **8.2** |
| Median variants/gene | 21 | 4 |
| Max variants/gene | 12,861 | 315 |

**M2TQA is 10.8x deeper** on average than MutaDescribe in variants per gene.
