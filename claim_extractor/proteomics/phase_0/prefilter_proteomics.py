"""
Prefilter module for protein extraction.

Filters out proteins and identifies those that are likely
being functionally discussed vs just mentioned in passing.
Adapted from phase_0/prefilter.py for proteomics use case.
"""

import re
from typing import Dict, List, Tuple, Set

# Proteomics-specific consequence/discussion patterns
PROTEOMICS_PATTERNS = [
    # Characterization verbs
    r"(characteriz|purif|identif|isolat|describ|determin)(ed|ing|es|ation)",
    r"(novel|new|first|unknown)\s+(protein|enzyme|factor)",

    # Enzymatic activity
    r"(catalyz|hydroly|cleav|digest|degrad)(ed|es|ing|ation)",
    r"(active\s+site|catalytic\s+(domain|residue|activity))",
    r"(substrate|product|cofactor|inhibitor)\s+(of|for|binding)",
    r"(k[_]?m|k[_]?cat|v[_]?max|ic[_]?50|k[_]?i|k[_]?d)\s*(=|of|is|was|were)",
    r"(enzyme|protein)\s+(activity|kinetics|assay)",
    r"(specific\s+activity|turnover\s+number|michaelis)",

    # Binding and interactions
    r"(bind|interact|associat)(s|ed|ing|ion)?\s+(to|with)",
    r"(affinity|avidity)\s+(for|of|to)",
    r"(complex|dimer|oligomer|multimer)(ization|izes|ized|ing)?",
    r"(co-immunoprecipitat|pull[- ]?down|cross[- ]?link)(ed|ing|ion)?",

    # Post-translational modifications
    r"(phosphorylat|acetylat|glycosylat|ubiquitinat|sumoylat|methylat|nitrosylat)(ed|es|ing|ion)",
    r"(modified|modification)\s+(at|of|by)",
    r"(PTM|post[- ]?translational)",

    # Structural features
    r"(crystal\s+structure|3[- ]?D\s+structure|tertiary\s+structure)",
    r"(domain|motif|fold|helix|sheet|loop)\s+(of|in|contains)",
    r"(N[- ]?terminal|C[- ]?terminal|transmembrane)",
    r"(active\s+site|binding\s+site|catalytic\s+site)",

    # Localization and expression
    r"(localiz|express|secret|transloca)(ed|es|ing|ion|ation)\s+(in|to|at)",
    r"(subcellular|cellular|tissue)\s+(localization|distribution)",
    r"(upregulat|downregulat|overexpress|underexpress)(ed|ion)",

    # Functional roles
    r"(function|role|involvement)\s+(of|in|as)",
    r"(essential|required|necessary|critical|important)\s+(for|in|to)",
    r"(regulat|modulat|activat|inhibit)(es|ed|ing|ion)\s+(the\s+)?",
    r"(signaling|pathway|cascade|mechanism)",

    # Biomarker/clinical relevance
    r"(biomarker|marker|indicator)\s+(of|for)",
    r"(diagnostic|prognostic|therapeutic)\s+(value|potential|target)",
    r"(serum|plasma|blood)\s+(level|concentration|biomarker)",
    r"(elevated|decreased|increased|reduced)\s+(in|level|expression)",

    # Stability and properties
    r"(stability|half[- ]?life|degradation)\s+(of|in)",
    r"(thermostab|pH[- ]?stab|proteolytic\s+stab)(le|ility)",
    r"(molecular\s+weight|isoelectric\s+point|pI)\s+(of|is|was)",

    # Recombinant/engineered proteins
    r"(recombinant|expressed\s+in|produced\s+in)",
    r"(mutant|variant|isoform)\s+(of|show|exhibit)",
]

PROTEOMICS_REGEX = re.compile(r"|".join(PROTEOMICS_PATTERNS), re.IGNORECASE)

# Patterns for proteins that are just referenced but not discussed
JUST_MENTIONED_PATTERNS = [
    r"(using|with|according\s+to)\s+\w+\s+(protocol|method|kit|antibody)",
    r"(see|cf\.|compare|reviewed\s+in)",
    r"(previously\s+reported|well[- ]?known|established)",
    r"(marker|control|standard)\s+(for|of)",
]

JUST_MENTIONED_REGEX = re.compile(r"|".join(JUST_MENTIONED_PATTERNS), re.IGNORECASE)


def split_into_sentences(text: str) -> List[str]:
    """Simple sentence splitter with abbreviation handling."""
    # Protect common abbreviations from splitting
    text = re.sub(r"(\b[A-Z])\.", r"\1<DOT>", text)
    text = re.sub(r"(et al)\.", r"\1<DOT>", text)
    text = re.sub(r"(Fig|Tab|Ref|No|vs|Dr|Mr|Mrs|Ms|e\.g|i\.e|cf|ca|approx)\.",
                  r"\1<DOT>", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d)\.", r"\1<DOT>", text)
    # Protect decimal numbers
    text = re.sub(r"(\d)<DOT>(\d)", r"\1.\2", text)

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.replace("<DOT>", ".") for s in sentences]

    return [s.strip() for s in sentences if s.strip()]


def find_protein_sentence_indices(text: str, sentences: List[str], protein: str) -> Set[int]:
    """Find which sentence indices contain the protein name."""
    indices = set()

    # Build sentence position map
    pos = 0
    sentence_ranges = []
    for sent in sentences:
        idx = text.find(sent, pos)
        if idx == -1:
            idx = pos
        sentence_ranges.append((idx, idx + len(sent)))
        pos = idx + len(sent)

    # Find all occurrences of protein
    escaped = re.escape(protein)
    # Allow for variations like hyphenation
    pattern = escaped.replace(r"\ ", r"[\s-]")

    for match in re.finditer(pattern, text, re.IGNORECASE):
        prot_pos = match.start()
        for i, (start, end) in enumerate(sentence_ranges):
            if start <= prot_pos < end:
                indices.add(i)
                break

    return indices


def check_discussion_proximity(
    sentences: List[str],
    protein_sentence_indices: Set[int],
    window: int = 1
) -> Tuple[bool, List[int], List[str]]:
    """
    Check if protein sentences or nearby sentences contain discussion language.

    Returns:
        Tuple of (is_discussed, evidence_sentence_indices, evidence_texts)
    """
    evidence_sentences = []
    evidence_texts = []

    for sent_idx in protein_sentence_indices:
        start_idx = max(0, sent_idx - window)
        end_idx = min(len(sentences), sent_idx + window + 1)

        for check_idx in range(start_idx, end_idx):
            sent = sentences[check_idx]
            if PROTEOMICS_REGEX.search(sent):
                # Check it's not just a passing mention
                if not JUST_MENTIONED_REGEX.search(sent):
                    evidence_sentences.append(check_idx)
                    evidence_texts.append(sent)

    # Deduplicate while preserving order
    unique_indices = sorted(set(evidence_sentences))
    unique_texts = []
    seen = set()
    for idx in unique_indices:
        text = sentences[idx]
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)

    return len(unique_indices) > 0, unique_indices, unique_texts[:3]


def is_substantive_protein(protein: str, text: str) -> bool:
    """
    Check if a protein name is substantive enough to be worth tracking.
    Filters out overly generic terms.
    """
    # Too short
    if len(protein) < 3:
        return False

    # Generic terms that aren't specific proteins
    generic_terms = {
        "protein", "enzyme", "receptor", "factor", "kinase",
        "antibody", "antigen", "marker", "biomarker",
        "target", "substrate", "inhibitor", "activator",
    }
    if protein.lower() in generic_terms:
        return False

    # Check if protein appears multiple times (suggests it's important)
    count = len(re.findall(re.escape(protein), text, re.IGNORECASE))
    if count < 2:
        return False

    return True


def prefilter_proteins(
    article_text: str,
    proteins: List[str],
    window: int = 1
) -> Dict:
    """
    Pre-filter proteins to identify those likely being discussed functionally.

    Args:
        article_text: Full text of the article
        proteins: List of candidate protein names
        window: Sentence window for context (default 1 = check adjacent sentences)

    Returns:
        Dict with:
        - likely_discussed: List of proteins with evidence
        - likely_just_mentioned: List of proteins without functional context
        - sentences: Numbered sentences for LLM verification
    """

    # Filter to substantive proteins
    substantive_proteins = [p for p in proteins if is_substantive_protein(p, article_text)]
    filtered_generic = [p for p in proteins if not is_substantive_protein(p, article_text)]

    sentences = split_into_sentences(article_text)

    likely_discussed = []
    likely_just_mentioned = []

    for protein in substantive_proteins:
        prot_sentences = find_protein_sentence_indices(article_text, sentences, protein)

        if not prot_sentences:
            likely_just_mentioned.append(protein)
            continue

        is_discussed, evidence_indices, evidence_texts = check_discussion_proximity(
            sentences, prot_sentences, window
        )

        if is_discussed:
            likely_discussed.append({
                "protein": protein,
                "protein_sentences": sorted(prot_sentences),
                "evidence_sentences": evidence_indices,
                "evidence_text": evidence_texts
            })
        else:
            likely_just_mentioned.append(protein)

    return {
        "likely_discussed": likely_discussed,
        "likely_just_mentioned": likely_just_mentioned,
        "filtered_generic": filtered_generic,
        "total_input": len(proteins),
        "total_substantive": len(substantive_proteins),
        "sentences": sentences,
    }


def build_gene_identifier_map(gene_annotations: List[Dict]) -> Dict[str, Dict]:
    """
    Build a map from gene symbol/mentions to identifiers.

    Returns a dict mapping protein name -> identifiers dict
    """
    id_map = {}

    for gene in gene_annotations:
        # Map by gene symbol
        symbol = gene.get("symbol", "")
        if symbol:
            id_map[symbol] = {
                "gene_id": gene.get("gene_id"),
                "taxon_id": gene.get("taxon_id"),
                "uniprot_accession": gene.get("uniprot_accession"),
                "uniprot_entry_name": gene.get("uniprot_entry_name"),
                "protein_name": gene.get("protein_name"),
            }

        # Also map by each mention
        for mention in gene.get("mentions", []):
            if mention and mention not in id_map:
                id_map[mention] = {
                    "gene_id": gene.get("gene_id"),
                    "taxon_id": gene.get("taxon_id"),
                    "uniprot_accession": gene.get("uniprot_accession"),
                    "uniprot_entry_name": gene.get("uniprot_entry_name"),
                    "protein_name": gene.get("protein_name"),
                }

    return id_map


def get_protein_names_from_annotations(gene_annotations: List[Dict]) -> List[str]:
    """Extract all protein/gene names from annotations."""
    names = set()

    for gene in gene_annotations:
        # Add gene symbol
        symbol = gene.get("symbol", "")
        if symbol:
            names.add(symbol)

        # Add all mentions
        for mention in gene.get("mentions", []):
            if mention:
                names.add(mention)

        # Add protein name if available
        prot_name = gene.get("protein_name", "")
        if prot_name:
            names.add(prot_name)

    return sorted(names)


def main():
    """Run the prefilter on extracted proteins."""
    import json
    from pathlib import Path

    # Load extracted proteins
    input_path = "./claim_extractor/proteomics/phase_0/output/extracted_proteins.json"
    output_path = "./claim_extractor/proteomics/phase_0/output/prefiltered_proteins.json"

    if not Path(input_path).exists():
        print(f"Error: {input_path} not found. Run extract_proteins.py first.")
        return

    print(f"Loading extracted proteins from {input_path}")
    with open(input_path, "r") as f:
        data = json.load(f)

    # Get protein mappings for enrichment
    protein_mappings = data.get("protein_mappings", {})

    results = {}
    total_discussed = 0
    total_mentioned = 0
    total_with_uniprot = 0

    for pmid, article in data["articles"].items():
        gene_annotations = article.get("gene_annotations", [])
        article_text = article.get("article_text", "")

        if not article_text:
            print(f"  PMID {pmid}: No article text, skipping")
            continue

        # Get protein names from PubTator annotations
        proteins = get_protein_names_from_annotations(gene_annotations)

        # Build identifier map
        id_map = build_gene_identifier_map(gene_annotations)

        result = prefilter_proteins(article_text, proteins, window=1)

        # Enrich discussed proteins with identifiers
        enriched_discussed = []
        for p in result["likely_discussed"]:
            protein_name = p["protein"]
            identifiers = id_map.get(protein_name, {})

            has_uniprot = bool(identifiers.get("uniprot_accession"))
            if has_uniprot:
                total_with_uniprot += 1

            enriched_discussed.append({
                **p,
                "gene_id": identifiers.get("gene_id"),
                "taxon_id": identifiers.get("taxon_id"),
                "uniprot_accession": identifiers.get("uniprot_accession"),
                "uniprot_entry_name": identifiers.get("uniprot_entry_name"),
                "protein_name_canonical": identifiers.get("protein_name"),
            })

        results[pmid] = {
            "title": article.get("title", ""),
            "likely_discussed": enriched_discussed,
            "likely_just_mentioned": result["likely_just_mentioned"],
            "filtered_generic": result["filtered_generic"],
            "gene_annotations": gene_annotations,  # Keep original annotations
            "stats": {
                "total_input": result["total_input"],
                "total_substantive": result["total_substantive"],
                "n_discussed": len(result["likely_discussed"]),
                "n_just_mentioned": len(result["likely_just_mentioned"]),
            },
            "sentences": result["sentences"],
        }

        n_discussed = len(result["likely_discussed"])
        n_mentioned = len(result["likely_just_mentioned"])
        total_discussed += n_discussed
        total_mentioned += n_mentioned

        print(f"  PMID {pmid}: {n_discussed} discussed, {n_mentioned} just mentioned")

    # Save results
    output = {
        "metadata": {
            "total_articles": len(results),
            "total_discussed": total_discussed,
            "total_just_mentioned": total_mentioned,
            "total_discussed_with_uniprot": total_with_uniprot,
        },
        "protein_mappings": protein_mappings,  # Carry forward
        "articles": results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nPrefilter complete!")
    print(f"  Total likely discussed: {total_discussed}")
    print(f"  Total with UniProt: {total_with_uniprot}")
    print(f"  Total just mentioned: {total_mentioned}")
    print(f"  Output saved to: {output_path}")


if __name__ == "__main__":
    main()
