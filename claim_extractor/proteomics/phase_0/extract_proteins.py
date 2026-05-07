#!/usr/bin/env python3
"""
Protein Extraction for Proteomics Pipeline using PubTator Central

Uses PubTator Central BioC annotations to extract:
- Gene IDs (NCBI Gene)
- Taxon IDs (NCBI Taxonomy)
- Maps Gene ID → UniProt accession (canonical)

This provides high-quality, standardized protein identifiers.

Usage:
    python extract_proteins.py
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# Paths
INPUT_PATH = "./claim_extractor/proteomics/proteomics_dataset/all_articles.json"
OUTPUT_PATH = "./claim_extractor/proteomics/phase_0/output/extracted_proteins.json"

# API endpoints
PUBTATOR_BIOC_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
UNIPROT_ID_MAPPING_URL = "https://rest.uniprot.org/idmapping"
UNIPROT_UNIPROTKB_URL = "https://rest.uniprot.org/uniprotkb"

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between requests


@dataclass
class GeneAnnotation:
    """A gene/protein annotation from PubTator."""
    gene_id: str  # NCBI Gene ID
    symbol: str  # Gene symbol
    name: str  # Full name if available
    taxon_id: Optional[str] = None  # NCBI Taxonomy ID
    mentions: List[str] = field(default_factory=list)  # Text mentions
    offsets: List[Tuple[int, int]] = field(default_factory=list)  # Character offsets


@dataclass
class ProteinMapping:
    """Mapping from Gene ID to UniProt."""
    gene_id: str
    gene_symbol: str
    taxon_id: Optional[str]
    uniprot_accession: Optional[str] = None
    uniprot_entry_name: Optional[str] = None
    protein_name: Optional[str] = None


# ============================================================================
# PUBTATOR CENTRAL API
# ============================================================================

def fetch_pubtator_annotations(pmids: List[str]) -> Dict[str, Dict]:
    """
    Fetch PubTator Central BioC annotations for a list of PMIDs.

    Returns:
        Dict mapping PMID -> annotation data
    """
    results = {}

    # PubTator can handle batches, but let's be conservative
    batch_size = 10

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        pmid_str = ",".join(batch)

        url = f"{PUBTATOR_BIOC_URL}?pmids={pmid_str}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            # PubTator3 API returns {"PubTator3": [...]} format
            if isinstance(data, dict) and "PubTator3" in data:
                docs = data["PubTator3"]
            elif isinstance(data, dict):
                docs = [data]
            else:
                docs = data

            for doc in docs:
                # Extract PMID from various possible fields
                pmid = doc.get("id") or doc.get("pmid") or ""
                if not pmid:
                    # Try to extract from _id field (format: "PMID|None")
                    _id = doc.get("_id", "")
                    if "|" in _id:
                        pmid = _id.split("|")[0]

                if pmid:
                    results[str(pmid)] = doc

        except requests.exceptions.RequestException as e:
            print(f"  Error fetching PubTator for batch {i}: {e}")

        time.sleep(REQUEST_DELAY)

    return results


def extract_gene_annotations(pubtator_doc: Dict) -> List[GeneAnnotation]:
    """
    Extract gene/protein annotations from a PubTator BioC document.

    Filters for:
    - Type: Gene (which includes proteins in PubTator)
    - Has NCBI Gene ID
    """
    gene_map = {}  # gene_id -> GeneAnnotation

    passages = pubtator_doc.get("passages", [])

    for passage in passages:
        for annot in passage.get("annotations", []):
            # Check if it's a Gene annotation
            infons = annot.get("infons", {})
            annot_type = infons.get("type", "")

            if annot_type.lower() != "gene":
                continue

            # Get the Gene ID (NCBI Gene)
            gene_id = infons.get("identifier") or infons.get("ncbi_gene") or ""

            # Skip if no gene ID
            if not gene_id:
                continue

            # Handle multiple IDs (sometimes comma-separated)
            gene_ids = [g.strip() for g in str(gene_id).split(",") if g.strip()]

            for gid in gene_ids:
                # Skip non-numeric IDs
                if not gid.replace(";", "").replace("-", "").isdigit():
                    continue

                # Get gene name from infons (more reliable than identifier)
                gene_name = infons.get("name", "")

                # Get other info
                symbol = gene_name or gid  # Use name as symbol if available
                if ";" in gid:
                    # Sometimes format is "gene_id;symbol"
                    parts = gid.split(";")
                    gid = parts[0]
                    if len(parts) > 1:
                        symbol = parts[1]

                # Get mention text
                text = annot.get("text", "")

                if gid not in gene_map:
                    gene_map[gid] = GeneAnnotation(
                        gene_id=gid,
                        symbol=symbol,
                        name=gene_name,
                        taxon_id=infons.get("ncbi_taxon") or infons.get("taxon_id"),
                        mentions=[],
                        offsets=[],
                    )
                # Update name if we have a better one
                elif gene_name and not gene_map[gid].name:
                    gene_map[gid].name = gene_name
                    gene_map[gid].symbol = gene_name

                if text and text not in gene_map[gid].mentions:
                    gene_map[gid].mentions.append(text)

                locations = annot.get("locations", [])
                for loc in locations:
                    offset = loc.get("offset", 0)
                    length = loc.get("length", 0)
                    gene_map[gid].offsets.append((offset, offset + length))

    return list(gene_map.values())


# ============================================================================
# UNIPROT ID MAPPING
# ============================================================================

def map_gene_ids_to_uniprot(gene_annotations: List[GeneAnnotation]) -> Dict[str, ProteinMapping]:
    """
    Map NCBI Gene IDs to UniProt accessions using UniProt ID mapping service.

    Returns:
        Dict mapping gene_id -> ProteinMapping
    """
    if not gene_annotations:
        return {}

    # Collect unique gene IDs
    gene_ids = list(set(g.gene_id for g in gene_annotations))

    # Build gene_id -> annotation map for metadata
    gene_to_annot = {g.gene_id: g for g in gene_annotations}

    # UniProt ID mapping in batches
    batch_size = 100
    results = {}

    for i in range(0, len(gene_ids), batch_size):
        batch = gene_ids[i:i + batch_size]

        try:
            # Submit ID mapping job
            job_id = submit_uniprot_mapping(batch)
            if not job_id:
                continue

            # Poll for results
            mappings = poll_uniprot_mapping(job_id)

            for gene_id, uniprot_data in mappings.items():
                annot = gene_to_annot.get(gene_id)
                results[gene_id] = ProteinMapping(
                    gene_id=gene_id,
                    gene_symbol=annot.symbol if annot else "",
                    taxon_id=annot.taxon_id if annot else None,
                    uniprot_accession=uniprot_data.get("accession"),
                    uniprot_entry_name=uniprot_data.get("entry_name"),
                    protein_name=uniprot_data.get("protein_name"),
                )

        except Exception as e:
            print(f"  Error mapping batch {i}: {e}")

        time.sleep(REQUEST_DELAY)

    # Add entries for unmapped genes
    for gene_id in gene_ids:
        if gene_id not in results:
            annot = gene_to_annot.get(gene_id)
            results[gene_id] = ProteinMapping(
                gene_id=gene_id,
                gene_symbol=annot.symbol if annot else "",
                taxon_id=annot.taxon_id if annot else None,
            )

    return results


def submit_uniprot_mapping(gene_ids: List[str]) -> Optional[str]:
    """Submit an ID mapping job to UniProt."""
    url = f"{UNIPROT_ID_MAPPING_URL}/run"

    data = {
        "from": "GeneID",
        "to": "UniProtKB",
        "ids": ",".join(gene_ids),
    }

    try:
        response = requests.post(url, data=data, timeout=30)
        response.raise_for_status()
        return response.json().get("jobId")
    except Exception as e:
        print(f"  Error submitting UniProt mapping: {e}")
        return None


def poll_uniprot_mapping(job_id: str, max_attempts: int = 20) -> Dict[str, Dict]:
    """Poll UniProt for mapping results."""
    results_url = f"{UNIPROT_ID_MAPPING_URL}/results/{job_id}"
    status_url = f"{UNIPROT_ID_MAPPING_URL}/status/{job_id}"

    for _ in range(max_attempts):
        try:
            # Check status - may contain results directly
            status_resp = requests.get(status_url, timeout=30)
            status_data = status_resp.json()

            # Results may be in the status response directly
            results_data = status_data

            # Or we may need to fetch from results endpoint
            if status_data.get("jobStatus") == "FINISHED" or "results" not in status_data:
                if status_data.get("jobStatus") == "RUNNING":
                    time.sleep(1)
                    continue

                response = requests.get(results_url, timeout=30)
                if response.status_code == 200:
                    results_data = response.json()

            if "results" in results_data:
                mappings = {}
                for result in results_data.get("results", []):
                    from_id = str(result.get("from", ""))
                    to_entry = result.get("to", {})

                    # Handle both string and dict formats
                    if isinstance(to_entry, str):
                        # Simple format: {"from": "123", "to": "P12345"}
                        if from_id not in mappings:  # Take first hit
                            mappings[from_id] = {
                                "accession": to_entry,
                                "entry_name": "",
                                "protein_name": "",
                            }
                    elif isinstance(to_entry, dict):
                        # Full format with details
                        accession = to_entry.get("primaryAccession", "")
                        entry_name = to_entry.get("uniProtkbId", "")
                        protein_name = ""

                        # Get protein name from proteinDescription
                        prot_desc = to_entry.get("proteinDescription", {})
                        rec_name = prot_desc.get("recommendedName", {})
                        if rec_name:
                            full_name = rec_name.get("fullName", {})
                            protein_name = full_name.get("value", "")

                        if from_id not in mappings:  # Take first hit
                            mappings[from_id] = {
                                "accession": accession,
                                "entry_name": entry_name,
                                "protein_name": protein_name,
                            }

                return mappings

            time.sleep(1)

        except Exception as e:
            print(f"  Error polling UniProt mapping: {e}")
            break

    return {}


def batch_lookup_uniprot(gene_ids: List[str]) -> Dict[str, Dict]:
    """
    Alternative: Direct lookup using UniProt search API.
    Useful as fallback if ID mapping service is slow or fails.
    """
    results = {}

    for gene_id in gene_ids:
        try:
            # Search UniProt for this gene ID using cross-reference
            url = f"{UNIPROT_UNIPROTKB_URL}/search"
            params = {
                "query": f"(xref:geneid-{gene_id})",
                "format": "json",
                "size": 5,  # Get a few results to pick reviewed if available
                "fields": "accession,id,protein_name,organism_id,reviewed",
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            entries = data.get("results", [])
            if entries:
                # Prefer reviewed (Swiss-Prot) entries
                reviewed = [e for e in entries if e.get("entryType", "").startswith("UniProtKB reviewed")]
                entry = reviewed[0] if reviewed else entries[0]

                prot_desc = entry.get("proteinDescription", {})
                rec_name = prot_desc.get("recommendedName", {})
                protein_name = rec_name.get("fullName", {}).get("value", "") if rec_name else ""

                results[gene_id] = {
                    "accession": entry.get("primaryAccession", ""),
                    "entry_name": entry.get("uniProtkbId", ""),
                    "protein_name": protein_name,
                }

        except Exception:
            pass  # Silent fail for individual lookups

        time.sleep(0.1)  # Be nice to the API

    return results


# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================

def extract_proteins_from_article(
    article: Dict,
    pubtator_doc: Optional[Dict]
) -> Dict:
    """Extract proteins from article using PubTator annotations."""

    pmid = article.get("pmid", "unknown")
    full_text = article.get("full_text", "")

    # Extract gene annotations from PubTator
    gene_annotations = []
    if pubtator_doc:
        gene_annotations = extract_gene_annotations(pubtator_doc)

    return {
        "pmid": pmid,
        "title": article.get("title", ""),
        "gene_annotations": [
            {
                "gene_id": g.gene_id,
                "symbol": g.symbol,
                "name": g.name,
                "taxon_id": g.taxon_id,
                "mentions": g.mentions,
            }
            for g in gene_annotations
        ],
        "article_text": full_text,
        "abstract": article.get("abstract", ""),
    }


def main():
    """Main extraction pipeline."""

    print(f"Loading articles from {INPUT_PATH}")
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    articles = data.get("articles", [])
    print(f"Found {len(articles)} articles to process")

    # Step 1: Collect all PMIDs
    pmids = [a.get("pmid") for a in articles if a.get("pmid")]
    print(f"\nStep 1: Fetching PubTator annotations for {len(pmids)} PMIDs...")

    pubtator_docs = fetch_pubtator_annotations(pmids)
    print(f"  Retrieved annotations for {len(pubtator_docs)} articles")

    # Step 2: Extract gene annotations from each article
    print("\nStep 2: Extracting gene annotations...")
    results = {}
    all_gene_annotations = []

    for article in articles:
        pmid = article.get("pmid", "unknown")
        pubtator_doc = pubtator_docs.get(pmid)

        result = extract_proteins_from_article(article, pubtator_doc)
        results[pmid] = result

        n_genes = len(result["gene_annotations"])
        all_gene_annotations.extend([
            GeneAnnotation(
                gene_id=g["gene_id"],
                symbol=g["symbol"],
                name=g.get("name", ""),
                taxon_id=g.get("taxon_id"),
                mentions=g.get("mentions", []),
            )
            for g in result["gene_annotations"]
        ])

        print(f"  PMID {pmid}: {n_genes} gene annotations")

    # Step 3: Map Gene IDs to UniProt
    unique_genes = {g.gene_id: g for g in all_gene_annotations}
    print(f"\nStep 3: Mapping {len(unique_genes)} unique Gene IDs to UniProt...")

    protein_mappings = map_gene_ids_to_uniprot(list(unique_genes.values()))
    mapped_count = sum(1 for m in protein_mappings.values() if m.uniprot_accession)
    print(f"  ID mapping service mapped {mapped_count} genes")

    # Fallback: Use UniProt search for unmapped genes
    unmapped_ids = [gid for gid, m in protein_mappings.items() if not m.uniprot_accession]
    if unmapped_ids:
        print(f"  Trying fallback lookup for {len(unmapped_ids)} unmapped genes...")
        fallback_results = batch_lookup_uniprot(unmapped_ids)

        for gene_id, data in fallback_results.items():
            if data.get("accession") and gene_id in protein_mappings:
                protein_mappings[gene_id].uniprot_accession = data["accession"]
                protein_mappings[gene_id].uniprot_entry_name = data.get("entry_name")
                protein_mappings[gene_id].protein_name = data.get("protein_name")

        fallback_count = sum(1 for m in protein_mappings.values() if m.uniprot_accession)
        print(f"  After fallback: {fallback_count} genes mapped to UniProt")

    # Step 4: Enrich results with UniProt mappings
    print("\nStep 4: Enriching results with UniProt mappings...")
    for pmid, result in results.items():
        for gene in result["gene_annotations"]:
            gene_id = gene["gene_id"]
            if gene_id in protein_mappings:
                mapping = protein_mappings[gene_id]
                gene["uniprot_accession"] = mapping.uniprot_accession
                gene["uniprot_entry_name"] = mapping.uniprot_entry_name
                gene["protein_name"] = mapping.protein_name

    # Calculate statistics
    total_genes = sum(len(r["gene_annotations"]) for r in results.values())
    total_with_uniprot = sum(
        1 for r in results.values()
        for g in r["gene_annotations"]
        if g.get("uniprot_accession")
    )
    final_mapped_count = sum(1 for m in protein_mappings.values() if m.uniprot_accession)

    # Save results
    output = {
        "metadata": {
            "total_articles": len(results),
            "articles_with_pubtator": len(pubtator_docs),
            "total_gene_annotations": total_genes,
            "unique_genes": len(unique_genes),
            "genes_mapped_to_uniprot": final_mapped_count,
            "total_annotations_with_uniprot": total_with_uniprot,
        },
        "protein_mappings": {
            gene_id: asdict(mapping)
            for gene_id, mapping in protein_mappings.items()
        },
        "articles": results,
    }

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("Extraction complete!")
    print(f"  Total articles: {len(results)}")
    print(f"  Articles with PubTator data: {len(pubtator_docs)}")
    print(f"  Total gene annotations: {total_genes}")
    print(f"  Unique genes: {len(unique_genes)}")
    print(f"  Mapped to UniProt: {final_mapped_count}")
    print(f"  Annotations with UniProt: {total_with_uniprot}")
    print(f"  Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
