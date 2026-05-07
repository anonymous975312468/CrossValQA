#!/usr/bin/env python3
"""
Map NCBI Gene IDs to UniProt accessions for multiple species.

Downloads ID mapping files from UniProt FTP for major model organisms
and maps all anchored variants to UniProt accessions.
"""

import gzip
import json
import os
import urllib.request
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Paths
CACHE_DIR = Path("./uniprot/input")
ANCHORED_FILE = Path("./pubtator/output/m2tqa_tiered_anchored_v3.jsonl")
OUTPUT_FILE = Path("./uniprot/output/variants_with_uniprot_all_species.jsonl")
STATS_FILE = Path("./uniprot/output/mapping_stats_all_species.json")

# UniProt FTP base URL
UNIPROT_FTP = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism"

# Species mapping: NCBI taxon ID -> (UniProt filename prefix, common name)
# Note: UniProt uses specific strain taxon IDs for some organisms
SPECIES_MAPPING = {
    "9606": ("HUMAN_9606", "Human"),
    "10090": ("MOUSE_10090", "Mouse"),
    "10116": ("RAT_10116", "Rat"),
    "7955": ("DANRE_7955", "Zebrafish"),
    "7227": ("DROME_7227", "Fruit fly"),
    "4932": ("YEAST_559292", "Yeast"),  # S. cerevisiae S288c
    "559292": ("YEAST_559292", "Yeast S288c"),
    "8355": ("XENLA_8355", "Xenopus laevis"),
    "9031": ("CHICK_9031", "Chicken"),
    "9615": ("CANLF_9615", "Dog"),
    "9913": ("BOVIN_9913", "Bovine"),
    "6239": ("CAEEL_6239", "C. elegans"),
    "3702": ("ARATH_3702", "Arabidopsis"),
    "9544": ("MACMU_9544", "Rhesus macaque"),
}


def download_mapping_file(species_prefix: str) -> Path:
    """Download UniProt ID mapping file for a species."""
    filename = f"{species_prefix}_idmapping.dat.gz"
    filepath = CACHE_DIR / filename
    url = f"{UNIPROT_FTP}/{filename}"

    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  Using cached: {filename} ({size_mb:.1f} MB)")
        return filepath

    print(f"  Downloading: {filename}")
    try:
        urllib.request.urlretrieve(url, filepath)
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"    Downloaded: {size_mb:.1f} MB")
        return filepath
    except Exception as e:
        print(f"    Failed to download: {e}")
        return None


def load_gene_to_uniprot(filepath: Path) -> dict:
    """
    Load NCBI GeneID to UniProt mapping from file.

    Returns dict: gene_id -> list of UniProt accessions
    """
    gene_to_uniprot = defaultdict(list)

    with gzip.open(filepath, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue

            uniprot_acc, id_type, id_value = parts

            if id_type == "GeneID":
                gene_to_uniprot[id_value].append(uniprot_acc)

    return dict(gene_to_uniprot)


def main():
    print("=" * 60)
    print("Multi-Species NCBI Gene ID → UniProt Mapping")
    print("=" * 60)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Scan dataset to find which species we need
    print("\nScanning dataset for species...")
    taxon_counts = defaultdict(int)

    with open(ANCHORED_FILE) as f:
        for line in f:
            rec = json.loads(line)
            taxon = rec.get("ncbi_taxon_id")
            if taxon:
                taxon_counts[taxon] += 1

    print(f"  Found {len(taxon_counts)} unique taxa")

    # Step 2: Download mapping files for species we have
    print("\nDownloading UniProt ID mapping files...")
    species_mappings = {}  # taxon_id -> gene_to_uniprot dict

    for taxon_id, count in sorted(taxon_counts.items(), key=lambda x: -x[1]):
        if taxon_id not in SPECIES_MAPPING:
            continue

        prefix, name = SPECIES_MAPPING[taxon_id]
        print(f"\n{name} (taxon {taxon_id}): {count:,} records")

        filepath = download_mapping_file(prefix)
        if filepath and filepath.exists():
            print(f"  Loading mappings...")
            gene_map = load_gene_to_uniprot(filepath)
            species_mappings[taxon_id] = gene_map
            print(f"  Loaded {len(gene_map):,} gene IDs")

    # Also add yeast mapping for taxon 4932 if we have 559292
    if "559292" in species_mappings and "4932" not in species_mappings:
        species_mappings["4932"] = species_mappings["559292"]

    print(f"\nTotal species with mappings: {len(species_mappings)}")

    # Step 3: Map all variants
    print("\nMapping variants to UniProt...")

    stats = {
        "total_records": 0,
        "with_taxon": 0,
        "with_gene_id": 0,
        "mapped_to_uniprot": 0,
        "by_species": defaultdict(lambda: {"total": 0, "mapped": 0}),
        "no_taxon": 0,
        "unsupported_taxon": 0,
        "unique_genes_mapped": set(),
        "unique_uniprot_ids": set(),
    }

    mapped_records = []

    with open(ANCHORED_FILE) as f:
        for line in tqdm(f, desc="Processing"):
            record = json.loads(line)
            stats["total_records"] += 1

            taxon = record.get("ncbi_taxon_id")
            gene_id = record.get("anchor_gene_id")

            if not taxon:
                stats["no_taxon"] += 1
                continue

            stats["with_taxon"] += 1
            stats["by_species"][taxon]["total"] += 1

            if not gene_id:
                continue

            stats["with_gene_id"] += 1

            # Check if we have mapping for this species
            if taxon not in species_mappings:
                stats["unsupported_taxon"] += 1
                continue

            # Look up UniProt accessions
            gene_map = species_mappings[taxon]
            uniprot_ids = gene_map.get(gene_id, [])

            if uniprot_ids:
                stats["mapped_to_uniprot"] += 1
                stats["by_species"][taxon]["mapped"] += 1
                stats["unique_genes_mapped"].add(gene_id)
                stats["unique_uniprot_ids"].update(uniprot_ids)

                record["uniprot_ids"] = uniprot_ids
                record["uniprot_primary"] = uniprot_ids[0]
                mapped_records.append(record)

    # Convert sets to counts
    stats["unique_genes_mapped"] = len(stats["unique_genes_mapped"])
    stats["unique_uniprot_ids"] = len(stats["unique_uniprot_ids"])
    stats["by_species"] = dict(stats["by_species"])

    # Step 4: Write output
    print(f"\nWriting output: {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        for record in mapped_records:
            f.write(json.dumps(record) + "\n")

    print(f"Writing stats: {STATS_FILE}")
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

    # Summary
    print()
    print("=" * 60)
    print("MAPPING SUMMARY")
    print("=" * 60)
    print(f"Total records: {stats['total_records']:,}")
    print(f"With taxon ID: {stats['with_taxon']:,}")
    print(f"No taxon ID: {stats['no_taxon']:,}")
    print(f"With gene ID: {stats['with_gene_id']:,}")
    print(f"Mapped to UniProt: {stats['mapped_to_uniprot']:,}")
    print(f"Unique genes: {stats['unique_genes_mapped']:,}")
    print(f"Unique UniProt IDs: {stats['unique_uniprot_ids']:,}")

    print("\nBy species:")
    for taxon, counts in sorted(stats["by_species"].items(), key=lambda x: -x[1]["total"]):
        if counts["total"] > 0:
            name = SPECIES_MAPPING.get(taxon, ("", "Unknown"))[1]
            pct = counts["mapped"] / counts["total"] * 100 if counts["total"] > 0 else 0
            print(f"  {name} ({taxon}): {counts['mapped']:,}/{counts['total']:,} ({pct:.1f}%)")

    print()
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
