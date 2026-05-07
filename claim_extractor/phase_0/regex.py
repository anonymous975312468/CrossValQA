"""
Mutation Identifier Extraction - Stage 1 (v2)

Robustly extracts explicit mutation identifiers from biomedical paper text with
high precision and controlled recall.

Key improvements over v1:
1. Fixed preprocessing to not break numbers like G231T
2. Better gene_int pattern to capture PEX26intG231T
3. Better gene_indel pattern to capture both G255insT and T35insC
4. Proper normalization to uppercase to avoid duplicates (PEX26R98W vs Pex26pR98W)
5. Added gene_compound pattern for L153V/C861del
6. Proper gating for bare variants

Design principles:
- Stage 1 detects mutation strings only. It does not infer, normalize to HGVS, or bind to claims.
- Precision over recall: prefer missing mutations over false positives
- Bare tokens only survive if supported by gene-anchored versions
"""

from locale import normalize
import re
from collections import Counter
from typing import Dict, List, Set, Tuple
import json
import sys


# ============================================================================
# SECTION TRIMMING
# ============================================================================

KEEP_SECTIONS = {
    "ABSTRACT",
    "INTRODUCTION", 
    "BACKGROUND",
    "DISCUSSION",
    "CONCLUSION",
    "CONCLUSIONS",
    "OBJECTIVES",
    "METHODS",
    "MEASUREMENTS AND MAIN RESULTS",
    "RESULTS",
    "FINDINGS",
    "INTERPRETATION",
}

DROP_SECTIONS = {
    "REFERENCES",
    "BIBLIOGRAPHY",
    "ACKNOWLEDGMENTS",
    "ACKNOWLEDGEMENTS",
    "FUNDING",
    "AUTHOR CONTRIBUTIONS",
    "COMPETING INTERESTS",
    "CONFLICTS OF INTEREST",
    "SUPPLEMENTARY REFERENCES",
    "SUPPLEMENTARY MATERIAL",
    "SUPPLEMENTARY METHODS",
    "EXTENDED METHODS",
    "MATERIALS",
}

SECTION_SPLIT_RE = re.compile(
    r"\n("
    r"(?:[A-Z][A-Z0-9 /&()\-\.,]{2,})"
    r"|"
    r"(?:[A-Z][a-z][A-Za-z0-9 /&()\-\.,]{1,})"
    r")\n"
)


def normalize_heading(h: str) -> str:
    """Normalize section heading for comparison."""
    return re.sub(r"[^A-Z0-9 /&()\-]", "", h.upper()).strip()


def filter_article_sections(text: str) -> Tuple[str, bool]:
    """
    Filter article text to keep only relevant sections.
    
    Returns:
        Tuple of (filtered_text, did_split_successfully)
    """
    parts = SECTION_SPLIT_RE.split(text)
    if len(parts) < 3:
        return text, False

    kept: List[str] = []
    preamble = parts[0].strip()
    if preamble:
        kept.append(preamble)

    kept_any_section = False
    for i in range(1, len(parts) - 1, 2):
        heading = parts[i].strip()
        if len(heading) > 60:
            continue

        content = parts[i + 1].strip()
        if not heading:
            continue

        norm = normalize_heading(heading)
        if norm in DROP_SECTIONS:
            continue

        if norm in KEEP_SECTIONS:
            kept_any_section = True
            if content:
                kept.append(f"\n{heading}\n{content}")

    if not kept_any_section:
        return text, False

    out = "\n".join(kept).strip()
    if len(out) < 0.3 * len(text):
        return text, False

    return out, True


# ============================================================================
# TEXT CLEANUP
# ============================================================================

_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\b")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)


def strip_emails_and_urls(text: str) -> str:
    """Remove email addresses and URLs to reduce false positives."""
    text = _EMAIL_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    return text


def normalize_gene_token(s: str) -> str:
    """
    Normalize gene-prefixed tokens to canonical form.
    
    Transformations:
    - Remove whitespace/newlines
    - Remove trailing 'p' before AA change (Pex26pR98W -> PEX26R98W)
    - Uppercase everything
    
    Examples:
        Pex26pR98W -> PEX26R98W
        PEX26 intG231T -> PEX26INTG231T
        pex26pg255insT -> PEX26G255INST
    """
    s = re.sub(r"\s+", "", s)
    # Remove protein suffix 'p' when followed by mutation notation
    # Handles: Pex26pR98W, Pex26pG255insT, Pex26pintG231T
    s = re.sub(r"p(?=[A-Z]\d|int[ACGT])", "", s, flags=re.IGNORECASE)
    return s.upper()


# ============================================================================
# NON-HGVS EXTRACTION
# ============================================================================

def extract_nonhgvs_variants(text: str) -> Dict[str, List[str]]:
    """
    Extract non-HGVS mutation identifiers from text.
    
    Mutation types captured:
    - gene_aa: PEX26R98W (gene + amino acid change)
    - gene_int: PEX26intG231T (intronic mutation with gene prefix)
    - gene_indel: PEX26G255insT, PEX26T35insC (insertions/deletions with gene prefix)
    - gene_compound: PEX26L153V/C861del (compound mutations with gene prefix)
    - nt_sub: C292T (bare nucleotide substitution)
    - nt_indel: C861del (bare nucleotide indel)
    - compound: L153V/C861del (bare compound mutation)
    - bare_aa: R98W (bare amino acid change, gated by gene_aa presence)
    - bare_int: intG231T (bare intronic mutation)
    
    Returns:
        Dict mapping mutation type to list of mutations found
    """

    BLOCKLIST = {
        "HEK293T", "HEK293", "HEK293A", "HEK293S",
        "CHO1", "CHO2", 
        "COS1", "COS7",
        "HeLa1", "HeLa2",
        "NIH3T3",
        "U2OS1",
    }

    # ---- Preprocessing ----
    t = text.replace("\u0001", " ")
    
    # Split packed tokens where digit '3' appears between two letters
    # e.g., "C292T3R98W" -> "C292T R98W"
    # But preserve numbers like "G231T" (3 between digits)
    t = re.sub(r"(?<=[A-Za-z])3(?=[A-Za-z])", " ", t)

    # ---- Pattern definitions ----
    # Gene prefix: Letters followed by digits (PEX26, ABCD1, HSD17B4)
    gene_prefix = r"(?:[A-Za-z]{2,10}\d{0,5})"
    gene_prefix_p = rf"(?:{gene_prefix}p?)"  # Optional trailing 'p' for protein

    # GENE-ANCHORED PATTERNS (highest confidence)
    
    # Gene + amino acid change: PEX26R98W, Pex26pL44P, VWF-N528S (optional hyphen)
    pat_gene_aa = re.compile(
        rf"\b({gene_prefix_p})-?([A-Z])(\d{{1,5}})([A-Z])\b",
        re.IGNORECASE
    )

    # Gene + intronic mutation: PEX26intG231T
    pat_gene_int = re.compile(
        rf"\b({gene_prefix_p})(int)([ACGT])(\d{{1,7}})([ACGT])\b",
        re.IGNORECASE
    )

    # Gene + nucleotide indel: PEX26G255insT, PEX26T35insC, PEX26C861del
    pat_gene_indel = re.compile(
        rf"\b({gene_prefix_p})([ACGT])(\d{{1,7}})(del|ins[ACGT]+)\b",
        re.IGNORECASE
    )

    # Gene + compound allele: PEX26L153V/C861del
    pat_gene_compound = re.compile(
        rf"\b({gene_prefix_p})([A-Z])(\d{{1,5}})([A-Z])\s*/\s*([ACGT])(\d{{1,7}})(del|ins[ACGT]+)\b",
        re.IGNORECASE
    )

    # Gene + space + amino acid change: BRAF V600E, TP53 R175H
    # Gene must be >= 3 chars to reduce false positives
    pat_gene_space_aa = re.compile(
        r"\b([A-Z][A-Z0-9]{2,10})\s+([A-Z])(\d{1,5})([A-Z])\b"
    )

    pat_bare_sub_arrow = re.compile(r"\b(\d{1,7})([ACGT])\s*>\s*([ACGT])\b")
    bare_sub_arrow = {f"{m.group(1)}{m.group(2)}>{m.group(3)}" for m in pat_bare_sub_arrow.finditer(t)}



    # BARE PATTERNS (lower confidence, gated)
    
    pat_nt_sub = re.compile(r"(?<!c\.)(?<![A-Za-z])\b([ACGT])(\d{1,7})([ACGT])\b")
    pat_nt_indel = re.compile(r"\b([ACGT])(\d{1,7})(del|ins[ACGT]+)\b", re.IGNORECASE)
    pat_compound = re.compile(
        r"\b([A-Z])(\d{1,5})([A-Z])\s*/\s*([ACGT])(\d{1,7})(del|ins[ACGT]+)\b",
        re.IGNORECASE
    )
    pat_bare_aa = re.compile(r"\b([A-Z])(\d{1,5})([A-Z])\b")
    pat_bare_int = re.compile(r"\b(int)([ACGT])(\d{1,7})([ACGT])\b", re.IGNORECASE)

    # ---- Extraction ----
    
    # Gene-anchored extractions (normalize to uppercase)
    gene_aa_raw = {
        normalize_gene_token(m.group(0)) 
        for m in pat_gene_aa.finditer(t)
        if normalize_gene_token(m.group(0)) not in BLOCKLIST
    }

    # Add space-separated gene+AA matches (e.g. "BRAF V600E" -> "BRAFV600E")
    for m in pat_gene_space_aa.finditer(t):
        gene = m.group(1)
        token = normalize_gene_token(m.group(0))
        if token not in BLOCKLIST and len(gene) >= 3:
            gene_aa_raw.add(token)
    gene_int_raw = {normalize_gene_token(m.group(0)) for m in pat_gene_int.finditer(t)}
    gene_indel_raw = {normalize_gene_token(m.group(0)) for m in pat_gene_indel.finditer(t)}
    gene_compound_raw = {normalize_gene_token(m.group(0)) for m in pat_gene_compound.finditer(t)}

    # Bare extractions
    nt_sub_raw = sorted({m.group(0).upper() for m in pat_nt_sub.finditer(t)})
    bare_int_raw = {re.sub(r"\s+", "", m.group(0)).upper() for m in pat_bare_int.finditer(t)}
    compound_raw = {re.sub(r"\s+", "", m.group(0)).upper() for m in pat_compound.finditer(t)}

    # ---- Post-filters / Gating ----
    
    def get_mutation_suffix(token: str, pattern_type: str) -> str:
        """Extract mutation suffix from gene-anchored token."""
        if pattern_type == "aa":
            match = re.search(r"[A-Z]\d+[A-Z]$", token)
            return match.group(0) if match else ""
        elif pattern_type == "int":
            match = re.search(r"INT[ACGT]\d+[ACGT]$", token, re.IGNORECASE)
            return match.group(0).upper() if match else ""
        elif pattern_type == "indel":
            match = re.search(r"[ACGT]\d+(?:DEL|INS[ACGT]+)$", token, re.IGNORECASE)
            return match.group(0).upper() if match else ""
        return ""

    gene_aa_suffixes = {get_mutation_suffix(v, "aa") for v in gene_aa_raw}
    gene_int_suffixes = {get_mutation_suffix(v, "int") for v in gene_int_raw}
    gene_indel_suffixes = {get_mutation_suffix(v, "indel") for v in gene_indel_raw}

    # Count bare indel occurrences
    nt_indel_hits = [m.group(0).upper() for m in pat_nt_indel.finditer(t)]
    nt_indel_counts = Counter(nt_indel_hits)

    # Compound support for indel gating
    compound_support_strs = list(compound_raw) + list(gene_compound_raw)

    # Gate bare indels: keep if gene-anchored OR occurs 2+ times OR in compound
    nt_indel_filtered = []
    for v in sorted(set(nt_indel_hits)):
        if v in gene_indel_suffixes:
            nt_indel_filtered.append(v)
        elif nt_indel_counts[v] >= 2:
            nt_indel_filtered.append(v)
        elif any(v in c for c in compound_support_strs):
            nt_indel_filtered.append(v)

    # Gate bare AA: only if gene-anchored version exists
    bare_aa_filtered = []
    if gene_aa_suffixes:
        for m in pat_bare_aa.finditer(t):
            token = m.group(0).upper()
            if token in gene_aa_suffixes:
                bare_aa_filtered.append(token)
        bare_aa_filtered = sorted(set(bare_aa_filtered))

    # Gate bare intronic: only if gene-anchored version exists
    bare_int_filtered = []
    if gene_int_suffixes:
        for token in bare_int_raw:
            if token in gene_int_suffixes:
                bare_int_filtered.append(token)
        bare_int_filtered = sorted(set(bare_int_filtered))

    return {
        "gene_aa": sorted(gene_aa_raw),
        "gene_int": sorted(gene_int_raw),
        "gene_indel": sorted(gene_indel_raw),
        "gene_compound": sorted(gene_compound_raw),
        "nt_sub": nt_sub_raw,
        "nt_indel": sorted(set(nt_indel_filtered)),
        "compound": sorted(compound_raw),
        "bare_sub_arrow": sorted(bare_sub_arrow),
        "bare_aa": bare_aa_filtered,
        "bare_int": bare_int_filtered,
    }


# ============================================================================
# HGVS EXTRACTION
# ============================================================================

def extract_hgvs_variants(text: str) -> Dict[str, List[str]]:
    """
    Extract HGVS-format variants from text.
    """

    text = text.replace('þ', '+')

    # Anchors to avoid matching inside words
    c_prefix = r"(?<![A-Za-z0-9])c\.\s*"
    p_prefix = r"(?<![A-Za-z0-9])p\.\s*"

    # 1-letter frameshift: p.S47fs, p.F412fs
    pat_p_1_fs = re.compile(rf"{p_prefix}[A-Z]\d+fs\b")

    # 1-letter deletion: p.K814del
    pat_p_1_del = re.compile(rf"{p_prefix}[A-Z]\d+del\b")

    # Non-standard coding format: c.G2578A, c.A533C (nucleotide-position-nucleotide)
    pat_c_sub_alt = re.compile(
        rf"{c_prefix}[ACGT]\d+[ACGT]\b"
    )

    # === CODING DNA VARIANTS ===
    
    # Splice site / intronic: c.768+358C>T, c.859-9T>C, c.6729+5_+19del
    pat_c_splice = re.compile(
        rf"{c_prefix}\d+[+\-]\d+(?:_[+\-]?\d+)?(?:[ACGT]\s*>\s*[ACGT]|del[ACGT]*|ins[ACGT]+|dup[ACGT]*)?",
        re.IGNORECASE
    )
    
    # Substitutions: c.1040C>T
    pat_c_sub = re.compile(
        rf"{c_prefix}\d+(?:_\d+)?\s*[ACGT]\s*(?:>|→)\s*[ACGT]\b"
    )
    
    # Deletions: c.5923del, c.4248_4250del
    pat_c_del = re.compile(
        rf"{c_prefix}\d+(?:_\d+)?del[ACGT]*\b",
        re.IGNORECASE
    )
    
    # Duplications: c.247_250dup
    pat_c_dup = re.compile(
        rf"{c_prefix}\d+(?:_\d+)?dup[ACGT]*\b",
        re.IGNORECASE
    )
    
    # Insertions: c.123_124insACGT
    pat_c_ins = re.compile(
        rf"{c_prefix}\d+(?:_\d+)?ins[ACGT]+\b",
        re.IGNORECASE
    )

    pat_p_1_fsX = re.compile(rf"{p_prefix}[A-Z]\d+fsX\d+\b")

    pat_p_del_legacy = re.compile(rf"{p_prefix}del[A-Z]\d+\b")


    # === PROTEIN VARIANTS ===
    
    # Stop codons: p.W855*, p.R408*, p.Trp855*
    pat_p_stop = re.compile(
        rf"(?:{p_prefix}[A-Z][a-z]{{2}}\d+\*)"
        rf"|(?:{p_prefix}[A-Z]\d+\*)"
        rf"|(?:{p_prefix}[A-Z][a-z]{{2}}\d+X\b)"
        rf"|(?:{p_prefix}[A-Z]\d+X\b)"
    )
    
    # Complex alleles: p.[L541P;A1038V], p.[Y245*;V767D]
    pat_p_complex = re.compile(
        rf"{p_prefix}\[[^\]]+\]"
    )
    

    # 3-letter substitution: p.Gly12Asp, p.Arg97Ter
    pat_p_3_sub = re.compile(rf"{p_prefix}[A-Z][a-z]{{2}}\d+(?:[A-Z][a-z]{{2}}|Ter)\b")
    pat_p_3_fs = re.compile(rf"{p_prefix}[A-Z][a-z]{{2}}\d+[A-Z][a-z]{{2}}fs\*\d+\b")
    pat_p_3_del = re.compile(rf"{p_prefix}[A-Z][a-z]{{2}}\d+(?:_[A-Z][a-z]{{2}}\d+)?del\b")
    
    # 1-letter: p.G12D, p.R97W
    pat_p_1 = re.compile(rf"{p_prefix}[A-Z]\d+[A-Z]\b")

    # === rsIDs ===
    pat_rsid = re.compile(r"\brs\d+\b")

    # === EXTRACTION ===
    def normalize(s):
        return re.sub(r"\s+", "", s)

    c_splice = sorted({normalize(m.group(0)) for m in pat_c_splice.finditer(text)})
    c_sub_std = {normalize(m.group(0)) for m in pat_c_sub.finditer(text)}
    c_sub_alt = {normalize(m.group(0)) for m in pat_c_sub_alt.finditer(text)}
    c_sub = sorted(c_sub_std | c_sub_alt)
    c_del = sorted({normalize(m.group(0)) for m in pat_c_del.finditer(text)})
    c_dup = sorted({normalize(m.group(0)) for m in pat_c_dup.finditer(text)})
    c_ins = sorted({normalize(m.group(0)) for m in pat_c_ins.finditer(text)})

    p_stop = sorted({normalize(m.group(0)) for m in pat_p_stop.finditer(text)})
    p_complex = sorted({normalize(m.group(0)) for m in pat_p_complex.finditer(text)})

    p3_sub = {normalize(m.group(0)) for m in pat_p_3_sub.finditer(text)}
    p3_fs = {normalize(m.group(0)) for m in pat_p_3_fs.finditer(text)}
    p3_del = {normalize(m.group(0)) for m in pat_p_3_del.finditer(text)}
    p3 = sorted(p3_sub | p3_fs | p3_del)

    p1_fs = {normalize(m.group(0)) for m in pat_p_1_fs.finditer(text)}
    p1_del = {normalize(m.group(0)) for m in pat_p_1_del.finditer(text)}
    p1_fsX = {normalize(m.group(0)) for m in pat_p_1_fsX.finditer(text)}
    p1_del_legacy = {normalize(m.group(0)) for m in pat_p_del_legacy.finditer(text)}
    p1 = sorted({normalize(m.group(0)) for m in pat_p_1.finditer(text)} | p1_fs | p1_del | p1_fsX | p1_del_legacy)

    rsids = sorted(set(pat_rsid.findall(text)))

    return {
        "c_splice": c_splice,
        "c_sub": c_sub,
        "c_del": c_del,
        "c_dup": c_dup,
        "c_ins": c_ins,
        "p_stop": p_stop,
        "p_complex": p_complex,
        "p_3letter": p3,
        "p_1letter": p1,
        "rsids": rsids,
    }

# ============================================================================
# MAIN API
# ============================================================================

def extract_all_variants(text: str, apply_section_filter: bool = True) -> Dict:
    """
    Main entry point: extract all variants from article text.
    
    Args:
        text: Full article text
        apply_section_filter: Whether to filter to relevant sections (default: True)
    
    Returns:
        Dict with:
        - hgvs: HGVS-format variants
        - non_hgvs: Non-HGVS format variants
        - did_split: Whether section filtering was applied
        - trimmed_length: Length after filtering
        - original_length: Original text length
    """
    if apply_section_filter:
        trimmed_text, did_split = filter_article_sections(text)
    else:
        trimmed_text, did_split = text, False
    
    trimmed_text = strip_emails_and_urls(trimmed_text)
    
    return {
        "hgvs": extract_hgvs_variants(trimmed_text),
        "non_hgvs": extract_nonhgvs_variants(trimmed_text),
        "did_split": did_split,
        "trimmed_length": len(trimmed_text),
        "original_length": len(text),
    }

# ============================================================================
# LLM-READY OUTPUT
# ============================================================================

def get_mutations_for_llm(results: Dict) -> Dict[str, any]:
    """
    Generate a clean, deduplicated mutation list suitable for LLM context.
    
    Prioritizes gene-anchored versions and removes redundant bare variants.
    
    Returns:
        Dict with:
        - mutations: List of unique mutation identifiers
        - by_type: Mutations organized by type for reference
        - context_string: Pre-formatted string for LLM prompt injection
    """
    # Collect gene-anchored mutations (highest confidence)
    gene_anchored = set()
    gene_anchored.update(results["non_hgvs"].get("gene_aa", []))
    gene_anchored.update(results["non_hgvs"].get("gene_int", []))
    gene_anchored.update(results["non_hgvs"].get("gene_indel", []))
    gene_anchored.update(results["non_hgvs"].get("gene_compound", []))
    
    # Collect HGVS mutations
    hgvs = set()
    for key in ["c_splice", "c_sub", "c_del", "c_dup", "c_ins", "p_stop", "p_complex", "p_3letter", "p_1letter"]:
        hgvs.update(results["hgvs"].get(key, []))
    
    # Collect rsIDs separately
    rsids = set(results["hgvs"].get("rsids", []))
    
    # Collect bare substitutions with arrow (775G>C format)
    bare_sub_arrow = set(results["non_hgvs"].get("bare_sub_arrow", []))
    
    # For bare variants, only include if NOT already represented by gene-anchored
    # Extract suffixes from gene-anchored to check for redundancy
    gene_anchored_suffixes = set()
    for mut in gene_anchored:
        # Extract suffix patterns
        # PEX26R98W -> R98W
        aa_match = re.search(r"[A-Z]\d+[A-Z]$", mut)
        if aa_match:
            gene_anchored_suffixes.add(aa_match.group(0))
        # PEX26INTG231T -> INTG231T
        int_match = re.search(r"INT[ACGT]\d+[ACGT]$", mut)
        if int_match:
            gene_anchored_suffixes.add(int_match.group(0))
        # PEX26G255INST -> G255INST
        indel_match = re.search(r"[ACGT]\d+(?:DEL|INS[ACGT]+)$", mut)
        if indel_match:
            gene_anchored_suffixes.add(indel_match.group(0))
    
    # Nucleotide substitutions - exclude if they're part of intronic notation
    nt_subs = set()
    gene_int_positions = set()
    for mut in results["non_hgvs"].get("gene_int", []):
        # Extract position pattern from INTG231T -> G231T
        match = re.search(r"[ACGT]\d+[ACGT]$", mut.replace("INT", ""))
        if match:
            gene_int_positions.add(match.group(0))
    
    for nt in results["non_hgvs"].get("nt_sub", []):
        if nt not in gene_int_positions:
            nt_subs.add(nt)
    
    # Build organized output
    by_type = {
        "gene_anchored_protein": sorted([m for m in gene_anchored if re.search(r"[A-Z]\d+[A-Z]$", m) and "INT" not in m and "/" not in m and "INS" not in m and "DEL" not in m]),
        "gene_anchored_intronic": sorted([m for m in gene_anchored if "INT" in m]),
        "gene_anchored_indel": sorted([m for m in gene_anchored if re.search(r"(?:INS|DEL)", m) and "/" not in m]),
        "gene_anchored_compound": sorted([m for m in gene_anchored if "/" in m]),
        "hgvs_coding": sorted([m for m in hgvs if m.startswith("c.")]),
        "hgvs_protein": sorted([m for m in hgvs if m.startswith("p.") and not m.startswith("p.[")]),
        "hgvs_protein_complex": sorted([m for m in hgvs if m.startswith("p.[")]),
        "nucleotide_changes": sorted(nt_subs | bare_sub_arrow),
        "rsids": sorted(rsids),
    }
    
    # Build flat list (gene-anchored preferred, no redundant bare)
    all_mutations = sorted(gene_anchored | hgvs | nt_subs | bare_sub_arrow | rsids)
    
    # Build context string for LLM
    lines = ["The following mutations were identified in this paper:"]
    
    if by_type["gene_anchored_protein"]:
        lines.append(f"  Protein changes (gene-anchored): {', '.join(by_type['gene_anchored_protein'])}")
    if by_type["gene_anchored_intronic"]:
        lines.append(f"  Intronic mutations (gene-anchored): {', '.join(by_type['gene_anchored_intronic'])}")
    if by_type["gene_anchored_indel"]:
        lines.append(f"  Insertions/Deletions (gene-anchored): {', '.join(by_type['gene_anchored_indel'])}")
    if by_type["gene_anchored_compound"]:
        lines.append(f"  Compound mutations (gene-anchored): {', '.join(by_type['gene_anchored_compound'])}")
    if by_type["hgvs_coding"]:
        lines.append(f"  HGVS coding: {', '.join(by_type['hgvs_coding'])}")
    if by_type["hgvs_protein"]:
        lines.append(f"  HGVS protein: {', '.join(by_type['hgvs_protein'])}")
    if by_type["hgvs_protein_complex"]:
        lines.append(f"  HGVS complex alleles: {', '.join(by_type['hgvs_protein_complex'])}")
    if by_type["nucleotide_changes"]:
        lines.append(f"  Nucleotide changes: {', '.join(by_type['nucleotide_changes'])}")
    if by_type["rsids"]:
        lines.append(f"  dbSNP IDs: {', '.join(by_type['rsids'])}")
    
    return {
        "mutations": all_mutations,
        "by_type": by_type,
        "context_string": "\n".join(lines),
    }

def format_mutations_for_prompt(results: Dict) -> str:
    """
    Generate a formatted string of mutations for inclusion in an LLM prompt.
    
    This is the primary function to use when preparing context for an LLM.
    """
    llm_data = get_mutations_for_llm(results)
    return llm_data["context_string"]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def summarize_results(results: Dict) -> str:
    """Generate a human-readable summary of extraction results."""
    lines = []
    lines.append(f"Section split: {results['did_split']}")
    lines.append(f"Text length: {results['trimmed_length']} (orig: {results['original_length']})")
    lines.append("")
    
    lines.append("HGVS variants:")
    for k, v in results["hgvs"].items():
        if v:
            lines.append(f"  {k}: {', '.join(v)}")
    
    lines.append("")
    lines.append("Non-HGVS variants:")
    for k, v in results["non_hgvs"].items():
        if v:
            lines.append(f"  {k}: {', '.join(v)}")
    
    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import ARTICLES_FILE

    DATA_PATH = str(ARTICLES_FILE)
    PMID = "20335223"

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    entry = data.get(PMID)
    article_text = entry["article"]

    results = extract_all_variants(article_text, apply_section_filter=True)
    mutation_context = format_mutations_for_prompt(results)
    print(mutation_context)

    print("Gene-anchored:", results["non_hgvs"]["gene_aa"])
    print("Bare AA:", results["non_hgvs"]["bare_aa"])
    print("\nFull context:")
    print(get_mutations_for_llm(results)["context_string"])