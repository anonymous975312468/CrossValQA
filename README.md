# CrossValQA Pipeline

A framework for generating high-quality, citation-grounded question-answer pairs from scientific literature via information isolation and bidirectional cross-validation.

## Pipeline Architecture

```
Phase 0 (Gemini)     Phase 1 (Gemini)     Phase 2 (DeepSeek)     Phase 3 (Llama)      UniProt Grounding
Entity Extraction  → QA Generation      → Independent Answers  → Cross-Validation   → Gene-to-UniProt
                     Q, A₁, R₁             A₂, R₂                R₂→A₁? R₁→A₂?       Mapping
```

### Information Isolation

The core innovation is that **LLM₂ never sees LLM₁'s answer or rationale**. Each model answers independently from the source text, and a third model cross-validates by checking whether each rationale supports the other's answer. QA pairs where both directions validate are retained; the rest are flagged as potential hallucinations.

## Directory Structure

```
claim_extractor/          Core pipeline (Phases 0-3)
  phase_0/                  Entity extraction and verification
  phase_1/                  QA generation (Gemini)
  phase_2/                  Independent answer generation (DeepSeek)
  phase_3/                  Cross-validation judge (Llama)
  proteomics/               Proteomics domain adaptation
  docs/                     Pipeline documentation

uniprot/                  Gene-to-UniProt grounding
  scripts/                  Grounding pipeline scripts
```

## Final Dataset

```
uniprot/output/m2tqa_gene_corrected.jsonl
```

### Record Fields

| Field | Description |
|---|---|
| `pmid` | PubMed article ID |
| `mutation` | Variant identifier (HGVS notation) |
| `question` | Generated question |
| `question_type` | Category (function, pathogenicity, disease, etc.) |
| `phase1_answer` | Phase 1 answer (Gemini) |
| `phase1_rationale` | Phase 1 rationale with sentence citations |
| `phase2_answer` | Phase 2 answer (DeepSeek) |
| `phase2_rationale` | Phase 2 rationale with sentence citations |
| `r1_supports_a2` | Cross-validation: does R₁ support A₂? |
| `r2_supports_a1` | Cross-validation: does R₂ support A₁? |
| `r1_a2_reasoning` | Judge reasoning for R₁→A₂ |
| `r2_a1_reasoning` | Judge reasoning for R₂→A₁ |
| `cited_sentences` | Extracted evidence sentences from source article |
| `corrected_gene` | Gene symbol (after UniProt grounding) |
| `gene_confidence` | Confidence tier (gold, high, low) |
| `uniprot_primary` | UniProt accession |
| `grounding` | Cross-validation result |

## Models Used

| Phase | Model | Purpose |
|---|---|---|
| Phase 0 | Gemini 2.5 Flash | Entity verification |
| Phase 1 | Gemini 2.5 Flash | QA generation |
| Phase 2 | DeepSeek Chat | Independent answer generation |
| Phase 3 | Llama 3.1 8B Instruct | Cross-validation judge |
| Grounding | (rule-based) | Gene-to-UniProt mapping |

## Proteomics Adaptation

The pipeline generalizes to other scientific domains. A proteomics adaptation (`claim_extractor/proteomics/`) was built as a proof of concept, with entity extraction adapted for protein names and synonyms.

## Running the Pipeline

```bash
export OPENROUTER_API_KEY="your-key"

cd claim_extractor

# Phase 0: Extract and verify mutations
python phase_0/batch_extractor.py

# Phase 1: Generate QA pairs
python phase_1/phase1_generator_qa.py --workers 50

# Phase 2: Independent answers
python phase_2/phase2_generator.py --workers 50

# Phase 3: Cross-validate
python phase_3/phase3_judge.py --workers 50

# UniProt grounding
python ../uniprot/scripts/master_grounding_pipeline.py
```

## Environment

- Python 3.10+
- Key dependencies: pandas, tqdm, aiohttp, dotenv
- LLM API: OpenRouter
- Platform: Linux
