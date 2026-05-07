"""
Prefilter module for mutation extraction.

Filters out gene names and identifies mutations that are likely 
being functionally discussed vs just mentioned in passing.
"""

import re
from typing import Dict, List, Tuple, Set

# Gene name patterns
GENE_NAME_PATTERNS = [
    r"^[A-Z]{2,6}\d{1,2}[A-Z]$",        # HMG20A, SCN5A
    r"^[A-Z]{3,10}$",                    # DCLRE, SEMA, BRCA
    r"^[A-Z]{2,6}\d{1,2}$",              # TP53, BRCA1, BRCA2
    r"^[A-Z]{2,6}\d{1,3}[A-Z]{1,2}$",   # NDUFA12L, TMEM126B
    r"^C\d{1,2}orf\d+$",                 # C12orf65
    r"^FAM\d+[A-Z]?$",                   # FAM110C
    r"^LOC\d+$",                         # LOC12345
    r"^KIAA\d+$",                        # KIAA0196
]

GENE_NAME_COMPILED = [re.compile(p, re.IGNORECASE) for p in GENE_NAME_PATTERNS]

# 3-letter amino acid codes
AMINO_ACIDS_3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", 
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", 
    "THR", "TRP", "TYR", "VAL", "SEC", "PYL",
    "TER", "STOP"
}

# 1-letter amino acid codes
AMINO_ACIDS_1 = set("ACDEFGHIKLMNPQRSTVWY*")

# Consequence patterns
CONSEQUENCE_PATTERNS = [
    r"caus(e[sd]?|ing)\s+(the\s+)?(disease|disorder|phenotype|syndrome)",
    r"result(s|ed|ing)?\s+in\s+(a\s+)?(loss|gain|reduction|increase|absence)",
    r"lead(s|ing)?\s+to\s+(a\s+)?(loss|gain|reduction|increase|absence|impair)",
    r"(is|was|were|classified\s+as)\s+(likely\s+)?(pathogenic|benign|deleterious)",
    r"(pathogenic|disease[- ]causing)\s+(variant|mutation)",
    r"(abolish|disrupt|impair|reduc|increas|diminish|enhanc)(es|ed|ing)?\s+(the\s+)?(function|activity|expression|stability|binding)",
    r"loss[- ]of[- ]function",
    r"gain[- ]of[- ]function",
    r"dominant[- ]negative",
    r"haploinsufficienc",
    r"(mutation|variant)\s+(was\s+)?(shown|found|demonstrated|confirmed)\s+to",
    r"(functional|in\s+vitro|in\s+vivo)\s+(analysis|studies?|assays?)\s+(showed|revealed|demonstrated|confirmed)",
    r"patients?\s+(with|carrying|harboring)\s+(the\s+)?(mutation|variant)",
    r"(heterozygous|homozygous|compound)\s+(for|carriers?)",
    r"(protein|enzyme)\s+(is|was)\s+(unstable|misfolded|degraded|truncated|absent)",
    r"(reduced|increased|abolished|absent)\s+(protein|enzyme)\s+(level|activity|expression)",
]

CONSEQUENCE_REGEX = re.compile(r"|".join(CONSEQUENCE_PATTERNS), re.IGNORECASE)


def is_likely_mutation(variant: str) -> bool:
    """Check if a variant string looks like a mutation vs a gene name."""
    
    # === WHITELIST: Definite mutation patterns ===
    definite_mutation_patterns = [
        r"^c\.\d+",                        # c.123A>G
        r"^p\.[A-Z]",                       # p.Arg123Ter  
        r"^rs\d+$",                         # rs12345
        r"\d+[ACGT]>[ACGT]",                # 775G>C
        r"[A-Z]\d+fs",                      # R98fs
        r"\d+del",                          # 123del
        r"\d+ins",                          # 123ins
        r"\d+dup",                          # 123dup
        r"^IVS\d+",                         # IVS1+1G>A
    ]
    
    for pattern in definite_mutation_patterns:
        if re.search(pattern, variant, re.IGNORECASE):
            return True
    
    # Gene-anchored mutations with hyphen (VWF-N528S)
    if re.match(r"^[A-Z]{2,10}\d{0,2}-[A-Z]\d{2,5}[A-Z]$", variant):
        return True
    
    # 3-letter AA code + position + 1-letter AA (GLY2279E)
    match_3to1 = re.match(r"^([A-Z]{3})(\d{2,5})([A-Z])$", variant, re.IGNORECASE)
    if match_3to1:
        aa3, pos, aa1 = match_3to1.groups()
        if aa3.upper() in AMINO_ACIDS_3 and aa1.upper() in AMINO_ACIDS_1:
            return True
    
    # 3-letter AA code + position + 3-letter AA (Gly2279Glu)
    match_3to3 = re.match(r"^([A-Z]{3})(\d{2,5})([A-Z]{3})$", variant, re.IGNORECASE)
    if match_3to3:
        aa1, pos, aa2 = match_3to3.groups()
        if aa1.upper() in AMINO_ACIDS_3 and aa2.upper() in AMINO_ACIDS_3:
            return True
    
    # 1-letter AA + position + 1-letter AA (R98W)
    if re.match(r"^[A-Z]\d{2,5}[A-Z]$", variant):
        return True
    
    # === BLACKLIST: Gene name patterns ===
    for pattern in GENE_NAME_COMPILED:
        if pattern.match(variant):
            return False
    
    # General heuristic
    if re.match(r"^[A-Z]{2,}[A-Z0-9]*$", variant):
        return False
    
    return True


def split_into_sentences(text: str) -> List[str]:
    """Simple sentence splitter."""
    text = re.sub(r"(\b[A-Z])\.", r"\1<DOT>", text)
    text = re.sub(r"(et al)\.", r"\1<DOT>", text)
    text = re.sub(r"(Fig|Tab|Ref|No|vs|Dr|Mr|Mrs|Ms)\.", r"\1<DOT>", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d)\.", r"\1<DOT>", text)
    
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.replace("<DOT>", ".") for s in sentences]
    
    return [s.strip() for s in sentences if s.strip()]


# ============================================================================
# GARBAGE SENTENCE FILTER (shared by Phase 1 and Phase 2)
# ============================================================================

GARBAGE_PATTERNS = [
    r'^All rights reserved',
    r'^Accepted Article',
    r'^This article is protected',
    r'^Author Man',
    r'^Copyright',
    r'^\s*\d+\s*$',
    r'^\s*[A-Z]{1,5}\s*$',
    r'^Table\s+\d',
    r'^Figure\s+\d',
    r'^Supplementary',
    r'^References?\s*$',
    r'^Acknowledgment',
    r'^\d+\s+g\.\d+',
]
GARBAGE_RE = re.compile('|'.join(GARBAGE_PATTERNS), re.IGNORECASE)


def is_garbage_sentence(sent: str) -> bool:
    """Check if a sentence is garbage (boilerplate, too short, non-text)."""
    if len(sent) < 20:
        return True
    if GARBAGE_RE.search(sent):
        return True
    alnum = sum(c.isalpha() for c in sent)
    if len(sent) > 0 and alnum < len(sent) * 0.5:
        return True
    return False


def find_mutation_sentence_indices(text: str, sentences: List[str], mutation: str) -> Set[int]:
    """Find which sentence indices contain the mutation."""
    indices = set()
    
    pos = 0
    sentence_ranges = []
    for sent in sentences:
        idx = text.find(sent, pos)
        if idx == -1:
            idx = pos
        sentence_ranges.append((idx, idx + len(sent)))
        pos = idx + len(sent)
    
    escaped = re.escape(mutation)
    for match in re.finditer(escaped, text, re.IGNORECASE):
        mut_pos = match.start()
        for i, (start, end) in enumerate(sentence_ranges):
            if start <= mut_pos < end:
                indices.add(i)
                break
    
    return indices


def check_consequence_proximity(
    sentences: List[str],
    mutation_sentence_indices: Set[int],
    window: int = 1
) -> Tuple[bool, List[int]]:
    """Check if mutation sentences or nearby sentences contain consequence language."""
    evidence_sentences = []
    
    for sent_idx in mutation_sentence_indices:
        start_idx = max(0, sent_idx - window)
        end_idx = min(len(sentences), sent_idx + window + 1)
        
        for check_idx in range(start_idx, end_idx):
            sent = sentences[check_idx]
            if CONSEQUENCE_REGEX.search(sent):
                evidence_sentences.append(check_idx)
    
    return len(evidence_sentences) > 0, sorted(set(evidence_sentences))


def prefilter_mutations(
    article_text: str, 
    mutations: List[str],
    window: int = 1
) -> Dict[str, any]:
    """
    Pre-filter mutations to identify those likely being discussed functionally.
    
    Returns:
        Dict with:
        - likely_discussed: List of mutations with evidence
        - likely_just_mentioned: List of mutations without functional context
        - filtered_gene_names: List of items that appear to be gene names
        - sentences: Numbered sentences for LLM verification
    """
    
    actual_mutations = [m for m in mutations if is_likely_mutation(m)]
    filtered_gene_names = [m for m in mutations if not is_likely_mutation(m)]
    
    sentences = split_into_sentences(article_text)
    
    likely_discussed = []
    likely_just_mentioned = []
    
    for mutation in actual_mutations:
        mut_sentences = find_mutation_sentence_indices(article_text, sentences, mutation)
        
        if not mut_sentences:
            likely_just_mentioned.append(mutation)
            continue
            
        has_consequence, evidence = check_consequence_proximity(
            sentences, mut_sentences, window
        )
        
        if has_consequence:
            likely_discussed.append({
                "mutation": mutation,
                "mutation_sentences": sorted(mut_sentences),
                "evidence_sentences": evidence,
                "evidence_text": [sentences[i] for i in evidence[:3]]
            })
        else:
            likely_just_mentioned.append(mutation)
    
    return {
        "likely_discussed": likely_discussed,
        "likely_just_mentioned": likely_just_mentioned,
        "filtered_gene_names": filtered_gene_names,
        "total_input": len(mutations),
        "total_actual_mutations": len(actual_mutations),
        "sentences": sentences,
    }
