# Phase 0: Entity Extraction - Technical Report

## Executive Summary

Phase 0 is the foundational stage of the claim extraction pipeline that identifies and extracts genetic mutations (entities) from biomedical literature. It consists of four specialized components working in sequence: mutation regex extraction, prefiltering, LLM verification, and batch processing. The phase outputs verified mutations with functional context evidence for downstream processing.

---

## 1. Entity Detection Mechanisms

### 1.1 Mutation Extraction Overview

Phase 0 uses a **dual-strategy mutation detection** system implemented in `phase_0/regex.py`:

#### Strategy 1: Non-HGVS Format Detection

Detects gene-prefixed and bare mutations using 10 distinct pattern categories with hierarchical confidence levels:

**High-Confidence (Gene-Anchored):**
| Pattern | Description | Examples |
|---------|-------------|----------|
| `gene_aa` | Gene + amino acid change | PEX26R98W, Pex26pL44P |
| `gene_int` | Intronic mutations with gene prefix | PEX26intG231T |
| `gene_indel` | Insertions/deletions with gene prefix | PEX26G255insT, PEX26T35insC |
| `gene_compound` | Compound alleles | PEX26L153V/C861del |

**Low-Confidence (Bare Mutations, Gated by Gene-Anchored Presence):**
| Pattern | Description | Examples | Gating Rule |
|---------|-------------|----------|-------------|
| `nt_sub` | Nucleotide substitution | C292T | Retained if occurs 2+ times |
| `nt_indel` | Bare nucleotide indel | C861del | Retained if gene-anchored exists OR occurs 2+ times |
| `compound` | Bare compound | L153V/C861del | Retained with gene context |
| `bare_aa` | Bare amino acid change | R98W | Only if gene-anchored version exists |
| `bare_int` | Bare intronic | intG231T | Only if gene-anchored version exists |
| `bare_sub_arrow` | Arrow notation | 1234G>C | Context-dependent |

**Regex Pattern Examples:**
```python
gene_prefix = r"(?:[A-Za-z]{2,10}\d{0,5})"
gene_prefix_p = rf"(?:{gene_prefix}p?)"

pat_gene_aa = re.compile(
    rf"\b({gene_prefix_p})\s*([A-Z])(\d{{1,5}})([A-Z])\b",
    re.IGNORECASE
)

pat_gene_int = re.compile(
    rf"\b({gene_prefix_p})(int)([ACGT])(\d{{1,7}})([ACGT])\b",
    re.IGNORECASE
)

pat_gene_indel = re.compile(
    rf"\b({gene_prefix_p})([ACGT])(\d{{1,7}})(del|ins[ACGT]+)\b",
    re.IGNORECASE
)
```

#### Strategy 2: HGVS Format Detection

Detects standardized HGVS (Human Genome Variation Society) notation with 10 distinct categories:

**Coding DNA Variants (c. prefix):**
| Format | Pattern | Examples |
|--------|---------|----------|
| `c_splice` | Splice site variants | c.768+358C>T, c.859-9T>C |
| `c_sub` | Substitutions | c.1040C>T |
| `c_del` | Deletions | c.5923del, c.4248_4250del |
| `c_dup` | Duplications | c.247_250dup |
| `c_ins` | Insertions | c.123_124insACGT |

**Protein Variants (p. prefix):**
| Format | Pattern | Examples |
|--------|---------|----------|
| `p_stop` | Stop codons | p.W855*, p.R408* |
| `p_complex` | Complex alleles | p.[L541P;A1038V] |
| `p_3letter` | 3-letter amino acid codes | p.Gly12Asp |
| `p_1letter` | 1-letter amino acid codes | p.G12D, p.R97W |

**Variant IDs:**
| Format | Pattern | Examples |
|--------|---------|----------|
| `rsids` | dbSNP identifiers | rs12345 |

### 1.2 Text Preprocessing

Before pattern matching, text undergoes preprocessing:

```python
def preprocess_text(text: str) -> str:
    # Remove null bytes
    text = text.replace("\u0001", "")

    # Handle packed tokens: "C292T3R98W" → "C292T R98W"
    # Splits on '3' between letters but preserves "G231T" notation

    # Remove emails and URLs
    text = strip_emails_and_urls(text)

    return text
```

### 1.3 Filtering and Validation

**Blocklist - Cell Line Identifiers:**
```python
BLOCKLIST = {
    "HEK293T", "HEK293", "HEK293A", "HEK293S",
    "CHO1", "CHO2", "COS1", "COS7",
    "HeLa1", "HeLa2", "NIH3T3", "U2OS1",
}
```

**Bare Mutation Gating Logic:**
1. Gene-anchored mutations → always retained
2. Bare indels → retained only if:
   - Gene-anchored version exists, OR
   - Occurs 2+ times in text, OR
   - Appears in compound mutation context
3. Bare AA/intronic → only retained if corresponding gene-anchored versions exist elsewhere in document

**Normalization:**
```python
def normalize_gene_token(s: str) -> str:
    s = re.sub(r"\s+", "", s)  # Remove whitespace/newlines
    # Remove protein suffix 'p' before mutation
    s = re.sub(r"p(?=[A-Z]\d|int[ACGT])", "", s, flags=re.IGNORECASE)
    return s.upper()  # Canonical uppercase form

# Examples:
# "Pex26pR98W" → "PEX26R98W"
# "PEX26 intG231T" → "PEX26INTG231T"
```

---

## 2. Context Window Construction

### 2.1 Sentence Segmentation

Phase 0 uses a hierarchical sentence splitting strategy:

**Primary: spaCy-based Segmentation**
```python
def sentence_split_spacy(text: str) -> Optional[List[Sent]]:
    nlp = get_sentence_nlp()
    # Loads en_core_sci_sm (scientific) or en_core_web_sm (general)
    doc = nlp(text)
    # Extracts sentence boundaries using parser + sentencizer
```

**Fallback 1: Regex-based Split**
```python
_SENT_SPLIT_FALLBACK = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z0-9\(\[])")
# Splits on punctuation followed by capital letters/brackets
```

**Fallback 2: Newline-based Split**
```python
def sentence_split_newline_fallback(text: str) -> List[Sent]:
    parts = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
```

**Output Structure:**
```python
@dataclass
class Sent:
    sid: int        # Sentence ID (0-indexed)
    start_char: int # Character offset in original text
    end_char: int   # End character offset
    text: str       # Exact sentence text
```

### 2.2 Mechanistic Keyword Windowing

Context windows are built around sentences containing mechanistic keywords:

**Mechanistic Keywords:**
```python
MECH_KEYWORDS = [
    # Change direction
    "increase", "decrease", "reduc", "elevat", "abolish", "impair", "enhanc",
    "no change", "unchanged", "did not affect", "failed to",

    # Molecular functions
    "activity", "function", "signaling", "phosphory", "binding", "interact",

    # Protein properties
    "stability", "abundance", "degrad", "half-life", "localiz", "traffic",

    # Biochemical processes
    "process", "cleav", "secretion", "secreted",
]
```

**Window Selection Algorithm:**
```python
CONTEXT_WINDOW = 10  # Sentences before and after

def select_sentence_windows(sentences, keywords, window=CONTEXT_WINDOW):
    keep = [False] * len(sentences)
    for i, s in enumerate(sentences):
        low = s.text.lower()
        if any(k in low for k in keywords):
            # Select ±CONTEXT_WINDOW sentences around matches
            for j in range(max(0, i - window), min(len(sentences), i + window + 1)):
                keep[j] = True
    return [sentences[i] for i in range(len(sentences)) if keep[i]]
```

**Configuration Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `CONTEXT_WINDOW` | 10 | Sentences before/after mechanistic keywords |
| `PROMPT_MAX_CHARS` | 16,000 | Max total characters sent to LLM |
| `MAX_RAW_CHARS` | 2,000,000 | Hard cap on raw article text |

---

## 3. Evidence Sentence Selection

### 3.1 Prefiltering Stage

The prefilter module (`prefilter.py`) implements mutation-to-evidence mapping before LLM verification:

**Phase 1: Mutation Classification**
```python
def is_likely_mutation(variant: str) -> bool:
    # Whitelist: definite mutation patterns (c., p., rs-, etc.)
    # Blacklist: gene name patterns (DCLRE, SEMA, etc.)
    # Heuristic checks for amino acid patterns
```

**Phase 2: Mutation Location Detection**
```python
def find_mutation_sentence_indices(text: str, sentences: List[str],
                                   mutation: str) -> Set[int]:
    # Finds all sentences containing the mutation string
    # Returns set of sentence indices
```

**Phase 3: Consequence Proximity Analysis**
```python
CONSEQUENCE_PATTERNS = [
    r"caus(e[sd]?|ing)\s+(the\s+)?(disease|disorder|phenotype)",
    r"result(s|ed|ing)?\s+in\s+(a\s+)?(loss|gain|reduction|increase)",
    r"(is|was|were|classified\s+as)\s+(likely\s+)?(pathogenic|benign)",
    r"(abolish|disrupt|impair|reduc|increas|diminish|enhanc)(es|ed|ing)?",
    r"loss[- ]of[- ]function",
    r"gain[- ]of[- ]function",
    # ... 15+ patterns total
]

def check_consequence_proximity(sentences: List[str],
                               mutation_sentence_indices: Set[int],
                               window: int = 1) -> Tuple[bool, List[int]]:
    # Checks ±1 sentence window around mutation for consequence language
    # Returns (has_functional_context, evidence_sentence_indices)
```

**Prefilter Output Structure:**
```python
def prefilter_mutations(article_text: str, mutations: List[str],
                       window: int = 1) -> Dict[str, any]:
    return {
        "likely_discussed": [
            {
                "mutation": "p.Gly1046Arg",
                "mutation_sentences": [35, 52],
                "evidence_sentences": [34, 35, 36, 51, 52, 53],
                "evidence_text": ["The mutation...", "Analysis showed..."]
            }
        ],
        "likely_just_mentioned": [...],  # No functional context
        "filtered_gene_names": [...],     # Filtered out as gene names
        "sentences": [all_sentences],
    }
```

### 3.2 LLM-Based Evidence Verification

**Evidence Building for LLM:**
```python
def build_numbered_sentences(sentences: List[str],
                            relevant_indices: List[int],
                            context_window: int = 2) -> str:
    # Expands indices by ±2 sentences for context
    # Limits to 100 total sentences to avoid overflow
    # Returns numbered format:
    # [0] First sentence text
    # [1] Second sentence text
    # ...
```

**LLM Verification Prompt:**
```
Analyze the following mutations from a scientific paper...

MUTATIONS TO ANALYZE:
{mutations_list}

RELEVANT PAPER EXCERPTS (numbered sentences):
{numbered_sentences}

For each mutation, respond with:
- mutation: string
- is_discussed: bool
- consequence_summary: string (ONLY directly stated info, no inferences)
- evidence_sentences: [list of sentence numbers]
- confidence: high/medium/low
```

**Validation Constraints:**
- `consequence_summary` must contain ONLY information directly stated in cited sentences
- No inferences or logical deductions allowed
- Every claim must be traceable to specific evidence sentences

---

## 4. Article-Level Filtering

### 4.1 Section Filtering

**Kept Sections:**
```python
KEEP_SECTIONS = {
    "ABSTRACT", "INTRODUCTION", "BACKGROUND", "DISCUSSION",
    "CONCLUSION", "CONCLUSIONS", "OBJECTIVES", "METHODS",
    "MEASUREMENTS AND MAIN RESULTS", "RESULTS", "FINDINGS", "INTERPRETATION"
}
```

**Dropped Sections:**
```python
DROP_SECTIONS = {
    "REFERENCES", "BIBLIOGRAPHY", "ACKNOWLEDGMENTS",
    "FUNDING", "AUTHOR CONTRIBUTIONS", "COMPETING INTERESTS",
    "SUPPLEMENTARY MATERIAL", "SUPPLEMENTARY METHODS", "EXTENDED METHODS"
}
```

**Section Detection Regex:**
```python
SECTION_SPLIT_RE = re.compile(
    r"\n("
    r"(?:[A-Z][A-Z0-9 /&()\-\.,]{2,})"      # ALL CAPS style
    r"|"
    r"(?:[A-Z][a-z][A-Za-z0-9 /&()\-\.,]{1,})"  # Title Case style
    r")\n"
)
```

**Quality Thresholds:**
| Threshold | Value | Purpose |
|-----------|-------|---------|
| Minimum filtered ratio | 30% | Keep filtered sections only if ≥30% of original |
| Max heading length | 60 chars | Discard headings >60 chars (likely false positives) |
| Min sentences | 3 | Require ≥3 sentences after filtering |
| Min filtered chars | 3,000 | Minimum characters in filtered text |

---

## 5. Output Structure

### 5.1 Primary Output Format

**JSONL File** (`stage1_claims.jsonl`):
```json
{
  "pubmed_id": "31661308",
  "doc_id": "PMID:31661308",
  "status": "ok",
  "did_split": true,
  "filter_mode": "sections",
  "raw_char_len": 45000,
  "filtered_char_len": 35000,
  "text_for_segmentation_char_len": 35000,
  "num_sentences_total": 287,
  "num_sentences_used_in_prompt": 120,
  "model": "deepseek-chat",
  "claims": [
    {
      "claim_id": "C0001",
      "claim_text": "The p.Gly1046Arg mutation disrupts DNA-binding...",
      "entities": ["ZNF592", "p.Gly1046Arg", "DNA-binding"],
      "scope_hint": "in patients",
      "evidence_sids": [35, 52, 57],
      "evidence": [
        {
          "sid": 35,
          "start_char": 12450,
          "end_char": 12680,
          "text": "The mutation analysis revealed..."
        }
      ]
    }
  ]
}
```

### 5.2 Verified Mutations Output

**JSON File** (`verified_mutations.json`):
```json
{
  "20531441": {
    "verified_mutations": [
      {
        "mutation": "p.Gly1046Arg",
        "is_discussed": true,
        "consequence_summary": "The mutation disrupts DNA-binding properties...",
        "evidence_sentences": [1, 2, 52, 57],
        "confidence": "high"
      }
    ],
    "unverified_mutations": [
      {
        "mutation": "rs12345",
        "is_discussed": false,
        "consequence_summary": null,
        "evidence_sentences": [],
        "confidence": "low"
      }
    ],
    "filtered_gene_names": ["HMG20A", "PDE8A"],
    "just_mentioned": ["some_variant"],
    "error": null
  }
}
```

### 5.3 Extraction Statistics

**JSON File** (`extraction_stats.json`):
```json
{
  "total_articles": 114831,
  "total_mutations_extracted": 1539316,
  "avg_mutations_per_article": 17.83,
  "processing_time_seconds": 3600,
  "articles_per_second": 31.9,
  "mutation_count_distribution": {
    "1": 8500,
    "5": 12000,
    "10": 9000
  },
  "zero_mutation_pmids": [...],
  "error_pmids": [...]
}
```

---

## 6. Processing Pipeline Workflow

### 6.1 Complete Phase 0 Flow

```
INPUT: PubMed articles with full text
  │
  ▼
[1] SECTION FILTERING
    ├── Remove references, supplementary material
    └── Keep abstract, methods, results, discussion
  │
  ▼
[2] SENTENCE SEGMENTATION
    ├── spaCy tokenization (primary)
    ├── Regex fallback
    └── Newline fallback
    Output: Numbered sentences with character offsets
  │
  ▼
[3] MUTATION EXTRACTION (regex.py)
    ├── Non-HGVS patterns: gene-anchored + bare mutations
    ├── HGVS patterns: c. and p. notation
    ├── Apply blocklists
    ├── Normalize and deduplicate
    └── Output: ~1,539,316 mutations from 114,831 articles
  │
  ▼
[4] CONTEXT WINDOWING
    ├── Identify mechanistic keywords in sentences
    ├── Select ±10 sentence windows around keywords
    └── Build numbered sentence prompt (max 16,000 chars)
  │
  ▼
[5] PREFILTERING (prefilter.py)
    ├── Classify mutations (real vs gene names)
    ├── Find mutation locations in sentences
    ├── Check for functional consequence language
    └── Separate "discussed" from "just_mentioned"
  │
  ▼
[6] LLM VERIFICATION (verify_mutations.py)
    ├── Batch mutations (10 per LLM call)
    ├── Verify is_discussed status
    ├── Cite specific evidence sentences
    └── Generate consequence summaries (no inferences)
  │
  ▼
OUTPUT: verified_mutations.json
├── Per PMID: verified, unverified, filtered genes, just_mentioned
└── Per mutation: mutation string, is_discussed, evidence, confidence
```

---

## 7. Technical Parameters

### 7.1 Configuration Constants

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `CONTEXT_WINDOW` | 10 | Sentences before/after mechanistic keywords |
| `PROMPT_MAX_CHARS` | 16,000 | Max characters for LLM prompt |
| `MAX_RAW_CHARS` | 2,000,000 | Hard cap on raw article text |
| `MIN_SENTENCES` | 3 | Minimum sentences after filtering |
| `MIN_ARTICLE_CHARS` | 5,000 | Minimum article length for processing |
| `MAX_MUTATIONS` | 200 | Articles with >200 mutations excluded |
| `MUTATION_WINDOW` | 1 | Sentences to check around mutation for evidence |
| `MAX_EVIDENCE_SENTENCES` | 100 | Cap on numbered sentences in LLM prompt |
| `LLM_TEMPERATURE` | 0.0 | Deterministic outputs |
| `OUTPUT_MAX_TOKENS` | 1,200 | Max tokens in LLM response |
| `CHECKPOINT_INTERVAL` | 1,000 | Save checkpoint every N articles |

### 7.2 API Configuration

| Setting | Value |
|---------|-------|
| Endpoint | `https://openrouter.ai/api/v1/chat/completions` |
| Default Model | `deepseek-chat` |
| Verification Model | `anthropic/claude-3-haiku` |
| Timeout | 120 seconds per request |
| Retry Logic | 3 attempts with exponential backoff |

---

## 8. Key Data Structures

```python
@dataclass
class Sent:
    sid: int          # Sentence ID (0-indexed)
    start_char: int   # Start character offset
    end_char: int     # End character offset
    text: str         # Sentence text

@dataclass
class VerifiedMutation:
    mutation: str                      # Normalized mutation string
    is_discussed: bool                 # Whether functionally discussed
    consequence_summary: Optional[str] # Summary of functional impact
    evidence_sentences: List[int]      # Sentence indices as evidence
    confidence: str                    # "high", "medium", or "low"

@dataclass
class ArticleResult:
    pmid: str
    verified_mutations: List[VerifiedMutation]
    filtered_gene_names: List[str]
    just_mentioned: List[str]
    error: Optional[str]
```

---

## 9. Quality Assurance Mechanisms

### 9.1 Checkpoint System
- Saves intermediate results every 1,000 articles
- Enables resume functionality on interruption
- Prevents data loss during long processing runs

### 9.2 Failed Batch Logging
- Logs failed LLM calls to separate file for retry
- Records: PMID, mutations, error reason, timestamp

### 9.3 Error Handling
- 3 retry attempts for LLM API errors
- Exponential backoff for rate limiting (429 status)
- Separates results into verified/unverified/error states

---

## 10. Dataset Statistics

**Input:**
- 114,831 PubMed articles with full text

**After Mutation Extraction:**
- 86,317 articles with ≥1 mutation
- 1,539,316 total mutations extracted
- Average: 17.83 mutations per article

**After LLM Verification:**
- 33,901 articles with verified mutations
- 234,148 total verified mutations
- Average: 6.9 verified mutations per article

**Mutation Type Distribution:**
| Type | Approximate % |
|------|---------------|
| Bare nucleotide (nt_sub) | ~40% |
| HGVS c. variants | 15-20% |
| HGVS p. variants | 10-15% |
| Gene-anchored mutations | 5-10% |
| Other formats | 15-20% |

---

## 11. File Manifest

| File | Purpose |
|------|---------|
| `phase_0/generate_claims.py` | Main LLM-based claim extraction with spaCy integration |
| `phase_0/regex.py` | Comprehensive mutation extraction patterns (HGVS + non-HGVS) |
| `phase_0/prefilter.py` | Mutation classification and consequence filtering |
| `phase_0/verify_mutations.py` | LLM verification with parallelization and checkpointing |
| `phase_0/batch_extractor.py` | Batch processing wrapper for parallel extraction |

---

## 12. Design Decisions

### 12.1 Precision vs Recall Trade-offs
- **Gene-anchored mutations always included** (highest precision)
- **Bare mutations filtered by gating** (prevents false positives from gene names)
- **Cell line blocklist** prevents false mutations from cell line identifiers
- **Consequence language requirement** ensures functional context exists

### 12.2 Evidence Chain
Phase 0 implements **explicit evidence tracing**:
1. Mutation appears in sentence N
2. Mechanistic keywords trigger sentence window selection
3. LLM extracts claims with specific evidence sentence IDs
4. Prefilter checks for functional consequence language
5. LLM verification cites specific evidence sentences
6. Final output includes character offsets traceable to original text

### 12.3 Why Custom Regex Over NER
- Biomedical mutations require domain-specific pattern matching
- Mutations are not standard NER categories (PERSON, ORG, LOC, MISC)
- HGVS notation follows strict grammatical rules amenable to regex
- Custom patterns allow hierarchical confidence scoring
