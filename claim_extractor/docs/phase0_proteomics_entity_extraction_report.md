# Phase 0: Entity Extraction (Proteomics Pipeline) - Technical Report

## Executive Summary

Phase 0 of the proteomics pipeline implements a three-stage entity extraction system designed to identify proteins discussed with functional context in scientific literature. Unlike the mutation-focused claim_extractor pipeline, this phase uses **PubTator Central's biomedical NER annotations** combined with **regex-based functional context filtering** and **LLM verification** to extract high-confidence protein entities.

**Key Statistics:**
- Input: 22 articles across 5 proteomics subdomain categories
- Stage 1: 67 gene annotations → 65 with UniProt mappings (97%)
- Stage 2: 74 proteins marked "likely discussed", 7 "just mentioned"
- Stage 3: 71 verified proteins with structured metadata

---

## 1. Architecture Overview

### 1.1 Three-Stage Pipeline

| Stage | Component | Purpose | Output |
|-------|-----------|---------|--------|
| **1** | `extract_proteins.py` | PubTator NER + UniProt mapping | 67 gene annotations |
| **2** | `prefilter_proteomics.py` | Functional context filtering | 74 discussed proteins |
| **3** | `verify_proteomics.py` | LLM verification | 71 verified proteins |

### 1.2 Pipeline Flow

```
INPUT: PubMed articles (22 PMIDs)
  │
  ▼
[Stage 1] PROTEIN EXTRACTION
  ├── Fetch from PubTator Central BioC API
  ├── Extract Gene annotations (type="Gene")
  ├── Map NCBI Gene IDs → UniProt accessions
  └── Output: 67 genes, 65 mapped to UniProt (97%)
  │
  ▼
[Stage 2] PREFILTERING
  ├── Filter generic terms (protein, enzyme, factor)
  ├── Check mention frequency (≥2 required)
  ├── Search ±1 sentence window for functional language
  └── Output: 74 "likely discussed", 7 "just mentioned"
  │
  ▼
[Stage 3] LLM VERIFICATION
  ├── Batch proteins (5 per LLM call)
  ├── Provide numbered evidence sentences
  ├── Verify with consequence summaries
  └── Output: 71 verified proteins with identifiers
```

---

## 2. Stage 1: Protein Extraction from PubTator Central

### 2.1 Data Source & APIs

**PubTator Central BioC API:**
| Setting | Value |
|---------|-------|
| Endpoint | `https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson` |
| Format | BioC JSON |
| Batch Size | Up to 10 PMIDs per request |
| Rate Limiting | 0.5 seconds between requests |

**UniProt ID Mapping:**
| Setting | Value |
|---------|-------|
| Service | UniProt ID Mapping REST API |
| Mapping | NCBI Gene ID → UniProtKB accessions |
| Batch Size | 100 genes per job |
| Fallback | Direct search: `(xref:geneid-{gene_id})` |

### 2.2 Entity Detection Method

**PubTator BioC Structure:**
```
Document → Passages (title, abstract, body)
    → Annotations (entity type, identifier, text, location)
```

**Extraction Algorithm:**
```python
For each passage in document:
    For each annotation:
        If type == "Gene":
            gene_id = annotation.infons["identifier"] or infons["ncbi_gene"]
            If gene_id is numeric and not empty:
                Extract all mentions (text matches)
                Record character offsets (start, end)
                Build gene_id → GeneAnnotation map
                Deduplicate on gene_id (accumulate mentions)
```

**Gene Annotation Data Structure:**
```python
@dataclass
class GeneAnnotation:
    gene_id: str              # NCBI Gene ID (e.g., "6286")
    symbol: str               # Gene symbol (e.g., "S100P")
    name: str                 # Full name from PubTator
    taxon_id: Optional[str]   # NCBI Taxonomy ID
    mentions: List[str]       # Text variations (e.g., ["S100P", "S100-P"])
    offsets: List[Tuple]      # Character positions in text
```

### 2.3 UniProt Mapping Strategy

**Primary Method: ID Mapping Service (Async Job Queue)**
```
1. Submit Job:
   - from: "GeneID"
   - to: "UniProtKB"
   - ids: "6286,780,3569,..."

2. Poll Status:
   - Check job status
   - Wait for FINISHED state

3. Fetch Results:
   - Extract: primaryAccession, uniProtkbId, protein name
```

**Fallback Method: Direct UniProt Search**
```python
Query: (xref:geneid-{gene_id})
Format: json
Size: 5 results
Selection: Prefer reviewed (SwissProt) over unreviewed (TrEMBL)
```

**Success Metrics:**
- Primary mapping: 65/67 genes (97%)
- Unmapped: 2 genes (taxon_id handling issues)

### 2.4 Output Format

**extracted_proteins.json:**
```json
{
  "metadata": {
    "total_articles": 22,
    "articles_with_pubtator": 22,
    "total_gene_annotations": 67,
    "unique_genes": 67,
    "genes_mapped_to_uniprot": 65
  },
  "protein_mappings": {
    "6286": {
      "gene_id": "6286",
      "gene_symbol": "S100P",
      "taxon_id": null,
      "uniprot_accession": "P25815",
      "uniprot_entry_name": "S100P_HUMAN",
      "protein_name": "Protein S100-P"
    }
  },
  "articles": {
    "23844161": {
      "pmid": "23844161",
      "title": "...",
      "gene_annotations": [
        {
          "gene_id": "396901",
          "symbol": "ORM1",
          "name": "ORM1",
          "taxon_id": null,
          "mentions": ["alpha1-acid glycoprotein", "AGP", "ORM1"],
          "uniprot_accession": "F1SN68",
          "protein_name": "Alpha-1-acid glycoprotein"
        }
      ],
      "article_text": "...",
      "abstract": "..."
    }
  }
}
```

---

## 3. Stage 2: Prefiltering - Functional Context Detection

### 3.1 Objective

**Problem:** Many proteins are simply mentioned in literature without functional discussion.

**Solution:** Use domain-specific regex patterns to identify sentences discussing protein function.

**Filtering Categories:**
- ✓ **Likely Discussed** - Has functional context
- ✗ **Just Mentioned** - Passing reference only
- ✗ **Filtered Generic** - Overly broad terms (protein, enzyme, etc.)

### 3.2 Proteomics-Specific Regex Patterns

**70+ patterns across 7 categories:**

#### Characterization & Identification
```regex
(characteriz|purif|identif|isolat|describ|determin)(ed|ing|es|ation)
(novel|new|first|unknown)\s+(protein|enzyme|factor)
```

#### Enzymatic Activity & Kinetics
```regex
(catalyz|hydroly|cleav|digest|degrad)(ed|es|ing|ation)
(active\s+site|catalytic\s+(domain|residue|activity))
(substrate|product|cofactor|inhibitor)\s+(of|for|binding)
(k[_]?m|k[_]?cat|v[_]?max|ic[_]?50|k[_]?i|k[_]?d)\s*(=|of|is|was|were)
(enzyme|protein)\s+(activity|kinetics|assay)
(specific\s+activity|turnover\s+number|michaelis)
```

#### Protein-Protein Interactions
```regex
(bind|interact|associat)(s|ed|ing|ion)?\s+(to|with)
(affinity|avidity)\s+(for|of|to)
(complex|dimer|oligomer|multimer)(ization|izes|ized|ing)?
(co-immunoprecipitat|pull[- ]?down|cross[- ]?link)(ed|ing|ion)?
```

#### Post-Translational Modifications
```regex
(phosphorylat|acetylat|glycosylat|ubiquitinat|sumoylat|methylat|nitrosylat)(ed|es|ing|ion)
(modified|modification)\s+(at|of|by)
(PTM|post[- ]?translational)
```

#### Structural Features
```regex
(crystal\s+structure|3[- ]?D\s+structure|tertiary\s+structure)
(domain|motif|fold|helix|sheet|loop)\s+(of|in|contains)
(N[- ]?terminal|C[- ]?terminal|transmembrane)
(active\s+site|binding\s+site|catalytic\s+site)
```

#### Localization & Expression
```regex
(localiz|express|secret|transloca)(ed|es|ing|ion|ation)\s+(in|to|at)
(subcellular|cellular|tissue)\s+(localization|distribution)
(upregulat|downregulat|overexpress|underexpress)(ed|ion)
```

#### Biomarker & Clinical Relevance
```regex
(biomarker|marker|indicator)\s+(of|for)
(diagnostic|prognostic|therapeutic)\s+(value|potential|target)
(serum|plasma|blood)\s+(level|concentration|biomarker)
(elevated|decreased|increased|reduced)\s+(in|level|expression)
```

### 3.3 Filtering Algorithm

**Step 1: Substantivity Check**
```python
def is_substantive_protein(protein: str, text: str) -> bool:
    # Filter 1: Length (≥3 characters)
    if len(protein) < 3: return False

    # Filter 2: Generic terms (blacklist)
    generic_terms = {"protein", "enzyme", "receptor", "factor", "kinase",
                     "antibody", "antigen", "marker", "biomarker"}
    if protein.lower() in generic_terms: return False

    # Filter 3: Frequency (must appear ≥2 times)
    count = len(re.findall(re.escape(protein), text, re.IGNORECASE))
    if count < 2: return False

    return True
```

**Step 2: Sentence-Level Context Window (±1 sentence)**
```python
def check_discussion_proximity(sentences, protein_sentence_indices, window=1):
    for sent_idx in protein_sentence_indices:
        start_idx = max(0, sent_idx - window)
        end_idx = min(len(sentences), sent_idx + window + 1)

        for check_idx in range(start_idx, end_idx):
            sent = sentences[check_idx]
            if PROTEOMICS_REGEX.search(sent):
                if not JUST_MENTIONED_REGEX.search(sent):
                    evidence_sentences.append(check_idx)
```

**Step 3: Negative Patterns (Just Mentioned Filter)**
```regex
(using|with|according\s+to)\s+\w+\s+(protocol|method|kit|antibody)
(see|cf\.|compare|reviewed\s+in)
(previously\s+reported|well[- ]?known|established)
(marker|control|standard)\s+(for|of)
```

### 3.4 Output Format

**prefiltered_proteins.json:**
```json
{
  "23844161": {
    "title": "...",
    "likely_discussed": [
      {
        "protein": "AGP",
        "protein_sentences": [0, 5, 6, 7],
        "evidence_sentences": [0, 4, 5, 7],
        "evidence_text": [
          "Alpha-1-acid glycoprotein (AGP) is a remarkable serum protein...",
          "...characterized by its high sialic acid content..."
        ],
        "gene_id": "396901",
        "uniprot_accession": "F1SN68",
        "protein_name_canonical": "Alpha-1-acid glycoprotein"
      }
    ],
    "likely_just_mentioned": ["Albumin"],
    "filtered_generic": ["protein"],
    "stats": {
      "total_input": 10,
      "total_substantive": 8,
      "n_discussed": 2,
      "n_just_mentioned": 1
    }
  }
}
```

---

## 4. Stage 3: LLM Verification

### 4.1 Configuration

| Setting | Value |
|---------|-------|
| API Service | OpenRouter |
| Model | Google Gemini 2.0 Flash |
| Concurrency | 10 parallel requests |
| Batch Size | 5 proteins per LLM call |
| Temperature | 0.1 (deterministic) |
| Max Tokens | 2000 |

### 4.2 Prompts

**System Prompt:**
```
You are a scientific literature analyst specializing in proteomics and biochemistry.
Your task is to determine whether proteins are substantively discussed in research papers.

A protein is "substantively discussed" if the paper describes:
- Its enzymatic activity, kinetic parameters, or catalytic mechanism
- Its structure, domains, or post-translational modifications
- Its interactions with other proteins or molecules
- Its biological function, role in pathways, or cellular localization
- Experimental characterization (purification, mass spectrometry, crystallography)
- Its relevance as a biomarker, therapeutic target, or diagnostic indicator

A protein is NOT substantively discussed if it is:
- Simply listed in a table or figure legend without explanation
- Mentioned as a housekeeping/control gene
- Referenced only as part of methodology (e.g., "anti-X antibody was used")
- Mentioned in passing in the introduction as background

CRITICAL: Your consequence_summary must ONLY include information that is
DIRECTLY stated in the cited sentences. Do not make inferences or logical
deductions beyond what is explicitly written.
```

**User Prompt Template:**
```
Analyze the following proteins from a scientific paper and determine which
ones are substantively discussed.

PROTEINS TO ANALYZE:
- PELP1
- AGP
- [...]

RELEVANT PAPER EXCERPTS (numbered sentences):
[0] Alpha-1-acid glycoprotein (AGP) is a remarkable serum protein...
[4] There is extensive homology between the pig gene and the human genes...
[...]

For each protein, respond in this exact JSON format:
{
  "results": [
    {
      "protein": "<protein_name>",
      "is_discussed": true/false,
      "consequence_summary": "<brief summary using ONLY information directly stated>",
      "evidence_sentences": [<list of sentence numbers>],
      "confidence": "high/medium/low"
    }
  ]
}
```

### 4.3 Context Window Construction

```python
def build_numbered_sentences(sentences, relevant_indices, context_window=2):
    """
    Expand evidence sentence indices by ±context_window,
    then format with numbering for LLM
    """
    expanded = set()
    for idx in relevant_indices:
        for i in range(max(0, idx - context_window),
                      min(len(sentences), idx + context_window + 1)):
            expanded.add(i)

    # Cap at 100 sentences for token budget
    sorted_indices = sorted(expanded)[:100]

    # Format as [0] sentence text, [1] sentence text, ...
    lines = [f"[{idx}] {sentences[idx]}" for idx in sorted_indices]
    return "\n".join(lines)
```

### 4.4 Data Structures

```python
@dataclass
class VerifiedProtein:
    protein: str
    is_discussed: bool
    consequence_summary: Optional[str]
    evidence_sentences: List[int]
    confidence: str  # high, medium, low
    identifiers: Optional[ProteinIdentifiers] = None

@dataclass
class ProteinIdentifiers:
    gene_id: Optional[str]
    taxon_id: Optional[str]
    uniprot_accession: Optional[str]
    uniprot_entry_name: Optional[str]
    protein_name_canonical: Optional[str]
```

### 4.5 Output Format

**verified_proteins.json:**
```json
{
  "41465416": {
    "verified_proteins": [
      {
        "protein": "PELP1",
        "is_discussed": true,
        "consequence_summary": "PELP1 is a scaffolding protein with a unique amino acid composition that is highly conserved across species...",
        "evidence_sentences": [2, 3],
        "confidence": "high",
        "gene_id": "27043",
        "taxon_id": null,
        "uniprot_accession": "Q8IZL8",
        "uniprot_entry_name": "PELP1_HUMAN",
        "protein_name_canonical": "Proline-, glutamic acid- and leucine-rich protein 1"
      }
    ],
    "unverified_proteins": [],
    "just_mentioned": [],
    "error": null
  }
}
```

**verification_stats.json:**
```json
{
  "total_articles": 22,
  "articles_with_verified_proteins": 10,
  "total_verified_proteins": 71,
  "verified_with_identifiers": 57,
  "total_unverified_by_llm": 3,
  "total_just_mentioned": 7,
  "processing_time_seconds": 19.15,
  "model": "google/gemini-2.0-flash-001"
}
```

---

## 5. Comparison: Proteomics vs. Mutations Pipeline

### 5.1 Entity Detection

| Aspect | Mutations Pipeline | Proteomics Pipeline |
|--------|-------------------|---------------------|
| **Entity Type** | Genetic mutations (c.123A>G, p.R98W) | Proteins (gene symbols, UniProt IDs) |
| **Detection Method** | Custom regex patterns | PubTator Central NER |
| **Database Mapping** | Downstream (post-validation) | Immediate (Stage 1) |
| **Precision Strategy** | Hierarchical gating | NER confidence + filtering |

### 5.2 Regex Patterns

**Mutations Pipeline:**
```python
# Highly constrained syntax (structural patterns)
pat_gene_aa = r"(PEX26)[A-Z](\d+)[A-Z]"           # PEX26R98W
pat_c_sub = r"c\.(\d+)[ACGT]>[ACGT]"              # c.1040C>T
pat_p_stop = r"p\.([A-Z]\d+\*)"                   # p.W855*
```

**Proteomics Pipeline:**
```python
# Broad semantic patterns (functional language)
r"(k[_]?m|k[_]?cat|v[_]?max)\s*(=|of|is)"        # Kinetic parameters
r"(phosphorylat|acetylat)(ed|ing|ion)"            # PTM discussion
r"(function|role|involvement)\s+(of|in|as)"       # Function discussion
```

### 5.3 Filtering Logic

| Aspect | Mutations | Proteins |
|--------|-----------|----------|
| **Substantivity** | Mutation vs gene name | Generic term filter + frequency |
| **Context Language** | "loss-of-function", "pathogenic" | "phosphorylation", "activity", "binding" |
| **Window Size** | ±1 sentence | ±1 sentence |

### 5.4 LLM Verification

| Aspect | Mutations | Proteins |
|--------|-----------|----------|
| **Question** | "Is this mutation pathogenic?" | "Is this protein discussed?" |
| **Context Window** | ±5 sentences | ±2 sentences |
| **Output** | pathogenic, consequence, confidence | is_discussed, summary, confidence |

### 5.5 Database Integration

**Mutations Pipeline:**
```
PMID → Extract mutations (regex) → Prefilter → Verify →
Gene recovery (external DB lookup)
```

**Proteomics Pipeline:**
```
PMID → PubTator NER → UniProt mapping (Stage 1) →
Prefilter → Verify → Ready for downstream
```

**Key Difference:** Proteomics has upstream database integration; mutations rely on downstream recovery.

---

## 6. Quality Metrics

### 6.1 Stage-by-Stage Preservation

| Stage | Input | Output | Preservation |
|-------|-------|--------|--------------|
| 1 (Extract) | 22 articles | 67 genes | 100% |
| 1 (Map) | 67 genes | 65 mapped | 97% |
| 2 (Prefilter) | 67 proteins | 74 discussed | 110%* |
| 3 (Verify) | 74 discussed | 71 verified | 96% |

*Some proteins have multiple mentions across articles.

### 6.2 Identifier Completeness

| Metric | Count | Percentage |
|--------|-------|------------|
| Verified proteins | 71 | 100% |
| With UniProt accession | 57 | 80.3% |
| With gene_id | 71 | 100% |
| With protein name | 57 | 80.3% |

### 6.3 Processing Performance

| Stage | Time | Rate |
|-------|------|------|
| Extract | ~2-3 min | 0.5s/batch |
| Prefilter | <1 sec | In-memory |
| Verify | ~20 sec | 10 concurrent |
| **Total** | ~3 min | 7 articles/min |

---

## 7. Configuration Parameters

### 7.1 File Paths

```python
INPUT_PATH = "proteomics/proteomics_dataset/all_articles.json"
EXTRACT_OUTPUT = "proteomics/phase_0/output/extracted_proteins.json"
PREFILTER_OUTPUT = "proteomics/phase_0/output/prefiltered_proteins.json"
VERIFY_OUTPUT = "proteomics/phase_0/output/verified_proteins.json"
```

### 7.2 API Configuration

| Service | Setting | Value |
|---------|---------|-------|
| PubTator | Rate limit | 0.5s between requests |
| UniProt | Batch size | 100 genes/job |
| OpenRouter | Timeout | 120 seconds |
| OpenRouter | Retries | 3 with exponential backoff |

### 7.3 Processing Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `context_window` | 2 | Sentences around evidence for LLM |
| `batch_size` | 5 | Proteins per LLM call |
| `max_concurrent` | 10 | Parallel LLM requests |
| `min_mentions` | 2 | Minimum protein occurrences |
| `max_sentences` | 100 | Cap on LLM context |

---

## 8. Error Handling

### 8.1 Stage 1 (Extract)

- Log failures to stdout
- Continue with available annotations
- Fallback to search API if mapping service fails

### 8.2 Stage 3 (Verify)

- Log failed batches to `failed_batches.jsonl`
- Retry with exponential backoff for rate limits
- Skip failed articles gracefully

**Failed Batch Format:**
```json
{
  "pmid": "23844161",
  "proteins": ["PELP1", "AGP"],
  "error": "parse_error: JSONDecodeError",
  "timestamp": "2025-01-28 14:22:15"
}
```

---

## 9. Design Rationale

### 9.1 Why PubTator Central?

1. **Trained NER Models** - Better alias handling than regex
2. **High Quality** - Full-text processing, not abstract-only
3. **Structured Output** - BioC format with identifiers
4. **UniProt Mapping** - Direct Gene ID extraction

### 9.2 Why Two-Stage Filtering?

1. **Efficiency** - Regex is fast (0.1s for all articles)
2. **Cost Reduction** - LLM only processes pre-filtered set
3. **Evidence Transparency** - LLM sees precomputed evidence

### 9.3 Why Consequence Summaries?

1. **Validation** - Must cite specific sentences
2. **Downstream Use** - QA generation in Phase 1
3. **Quality Verification** - Spot unsupported claims
4. **Information Flow** - Separates stated vs inferred

---

## 10. File Manifest

| File | Purpose |
|------|---------|
| `phase_0/extract_proteins.py` | PubTator NER + UniProt mapping |
| `phase_0/prefilter_proteomics.py` | Functional context regex filtering |
| `phase_0/verify_proteomics.py` | LLM verification with consequence summaries |
| `phase_0/output/extracted_proteins.json` | 67 genes with identifiers |
| `phase_0/output/prefiltered_proteins.json` | 74 discussed + 7 mentioned |
| `phase_0/output/verified_proteins.json` | 71 verified proteins |
| `phase_0/output/verification_stats.json` | Processing metrics |

---

## 11. Summary

The proteomics Phase 0 pipeline achieves:

- **97% identifier coverage** (65/67 genes mapped to UniProt)
- **96% LLM acceptance rate** (71/74 prefiltered proteins verified)
- **80% complete identifier enrichment** (57/71 with all fields)
- **~3 minute processing** for 22 articles

**Key differences from mutations pipeline:**
1. Uses **trained NER** (PubTator) instead of regex-only detection
2. **Immediate database mapping** instead of downstream recovery
3. **Broader functional filtering** (experimental evidence > genetic consequence)
4. **Simplified verification** (discussion existence > impact assertion)
