#!/usr/bin/env python3
"""
Map NCBI Gene IDs to UniProt accessions for multiple species.

Includes recovery of unlabeled records by checking if their gene IDs
exist in known species mappings (primarily human).
"""

import gzip
import json
import urllib.request
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Paths
CACHE_DIR = Path("./uniprot/input")
ANCHORED_FILE = Path("./pubtator/output/m2tqa_tiered_anchored_v3.jsonl")
OUTPUT_FILE = Path("./uniprot/output/variants_with_uniprot_recovered.jsonl")
STATS_FILE = Path("./uniprot/output/mapping_stats_recovered.json")

# UniProt FTP base URL
UNIPROT_FTP = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism"

# Species mapping: NCBI taxon ID -> (UniProt filename prefix, common name)
SPECIES_MAPPING = {
    "9606": ("HUMAN_9606", "Human"),
    "10090": ("MOUSE_10090", "Mouse"),
    "10116": ("RAT_10116", "Rat"),
    "7955": ("DANRE_7955", "Zebrafish"),
    "7227": ("DROME_7227", "Fruit fly"),
    "4932": ("YEAST_559292", "Yeast"),
    "559292": ("YEAST_559292", "Yeast S288c"),
    "6239": ("CAEEL_6239", "C. elegans"),
    "9031": ("CHICK_9031", "Chicken"),
}


def download_mapping_file(species_prefix: str) -> Path:
    """Download UniProt ID mapping file for a species."""
    filename = f"{species_prefix}_idmapping.dat.gz"
    filepath = CACHE_DIR / filename
    url = f"{UNIPROT_FTP}/{filename}"

    if filepath.exists():
        return filepath

    print(f"  Downloading: {filename}")
    try:
        urllib.request.urlretrieve(url, filepath)
        return filepath
    except Exception as e:
        print(f"    Failed: {e}")
        return None


def load_gene_to_uniprot(filepath: Path) -> dict:
    """Load NCBI GeneID to UniProt mapping from file."""
    gene_to_uniprot = defaultdict(list)

    with gzip.open(filepath, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3 and parts[1] == "GeneID":
                gene_to_uniprot[parts[2]].append(parts[0])

    return dict(gene_to_uniprot)


def main():
    print("=" * 60)
    print("Multi-Species UniProt Mapping with Recovery")
    print("=" * 60)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load all species mappings
    print("\nLoading species mappings...")
    species_mappings = {}  # taxon_id -> gene_to_uniprot dict
    all_gene_to_species = {}  # gene_id -> (taxon_id, uniprot_ids) for recovery

    for taxon_id, (prefix, name) in SPECIES_MAPPING.items():
        if taxon_id in species_mappings:
            continue

        filepath = CACHE_DIR / f"{prefix}_idmapping.dat.gz"
        if not filepath.exists():
            filepath = download_mapping_file(prefix)

        if filepath and filepath.exists():
            gene_map = load_gene_to_uniprot(filepath)
            species_mappings[taxon_id] = gene_map
            print(f"  {name}: {len(gene_map):,} gene IDs")

            # Build reverse lookup for recovery (human takes priority)
            if taxon_id == "9606":
                for gene_id, uniprot_ids in gene_map.items():
                    all_gene_to_species[gene_id] = (taxon_id, uniprot_ids)

    # Add other species to recovery (only if not already in human)
    for taxon_id, gene_map in species_mappings.items():
        if taxon_id == "9606":
            continue
        for gene_id, uniprot_ids in gene_map.items():
            if gene_id not in all_gene_to_species:
                all_gene_to_species[gene_id] = (taxon_id, uniprot_ids)

    # Yeast alias
    if "559292" in species_mappings and "4932" not in species_mappings:
        species_mappings["4932"] = species_mappings["559292"]

    print(f"\nTotal gene IDs for recovery lookup: {len(all_gene_to_species):,}")

    # Step 2: Map all variants with recovery
    print("\nMapping variants...")

    stats = {
        "total_records": 0,
        "with_taxon_mapped": 0,
        "no_taxon_recovered": 0,
        "no_taxon_unrecoverable": 0,
        "unsupported_taxon": 0,
        "no_gene_id": 0,
        "gene_not_in_mapping": 0,
        "by_species": defaultdict(lambda: {"total": 0, "mapped": 0}),
        "recovered_species": defaultdict(int),
    }

    mapped_records = []

    with open(ANCHORED_FILE) as f:
        for line in tqdm(f, desc="Processing"):
            record = json.loads(line)
            stats["total_records"] += 1

            gene_id = record.get("anchor_gene_id")
            if not gene_id:
                stats["no_gene_id"] += 1
                continue

            taxon = record.get("ncbi_taxon_id")

            # Case 1: Has taxon ID - use species-specific mapping
            if taxon:
                stats["by_species"][taxon]["total"] += 1

                if taxon in species_mappings:
                    uniprot_ids = species_mappings[taxon].get(gene_id, [])
                    if uniprot_ids:
                        stats["with_taxon_mapped"] += 1
                        stats["by_species"][taxon]["mapped"] += 1
                        record["uniprot_ids"] = uniprot_ids
                        record["uniprot_primary"] = uniprot_ids[0]
                        record["mapping_source"] = "direct"
                        mapped_records.append(record)
                    else:
                        stats["gene_not_in_mapping"] += 1
                else:
                    stats["unsupported_taxon"] += 1

            # Case 2: No taxon ID - try to recover using gene ID lookup
            else:
                if gene_id in all_gene_to_species:
                    recovered_taxon, uniprot_ids = all_gene_to_species[gene_id]
                    stats["no_taxon_recovered"] += 1
                    stats["recovered_species"][recovered_taxon] += 1

                    record["uniprot_ids"] = uniprot_ids
                    record["uniprot_primary"] = uniprot_ids[0]
                    record["ncbi_taxon_id"] = recovered_taxon  # Inferred taxon
                    record["mapping_source"] = "recovered"
                    mapped_records.append(record)
                else:
                    stats["no_taxon_unrecoverable"] += 1

    # Convert for JSON
    stats["by_species"] = dict(stats["by_species"])
    stats["recovered_species"] = dict(stats["recovered_species"])

    # Step 3: Write output
    print(f"\nWriting output: {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        for record in mapped_records:
            f.write(json.dumps(record) + "\n")

    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

    # Summary
    total_mapped = stats["with_taxon_mapped"] + stats["no_taxon_recovered"]

    print()
    print("=" * 60)
    print("MAPPING SUMMARY")
    print("=" * 60)
    print(f"Total records: {stats['total_records']:,}")
    print(f"No gene ID: {stats['no_gene_id']:,}")
    print()
    print(f"With taxon - mapped: {stats['with_taxon_mapped']:,}")
    print(f"With taxon - unsupported species: {stats['unsupported_taxon']:,}")
    print(f"With taxon - gene not found: {stats['gene_not_in_mapping']:,}")
    print()
    print(f"No taxon - RECOVERED: {stats['no_taxon_recovered']:,}")
    print(f"No taxon - unrecoverable: {stats['no_taxon_unrecoverable']:,}")
    print()
    print(f"TOTAL MAPPED: {total_mapped:,} ({total_mapped/stats['total_records']*100:.1f}%)")

    print("\nRecovered by inferred species:")
    for taxon, count in sorted(stats["recovered_species"].items(), key=lambda x: -x[1]):
        name = SPECIES_MAPPING.get(taxon, ("", "Unknown"))[1]
        print(f"  {name} ({taxon}): {count:,}")

    print("\nBy labeled species:")
    for taxon, counts in sorted(stats["by_species"].items(), key=lambda x: -x[1]["total"])[:10]:
        if counts["total"] > 0:
            name = SPECIES_MAPPING.get(taxon, ("", taxon))[1]
            pct = counts["mapped"] / counts["total"] * 100 if counts["total"] > 0 else 0
            print(f"  {name}: {counts['mapped']:,}/{counts['total']:,} ({pct:.1f}%)")

    print()
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
