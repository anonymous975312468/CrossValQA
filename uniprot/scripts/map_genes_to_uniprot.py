#!/usr/bin/env python3
"""
Map NCBI Gene IDs to UniProt accessions using UniProt ID mapping file.

Downloads the human ID mapping from UniProt FTP and joins with anchored variants.
"""

import gzip
import json
import os
import urllib.request
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Paths
MAPPING_URL = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz"
CACHE_DIR = Path("./uniprot/input")
MAPPING_FILE = CACHE_DIR / "HUMAN_9606_idmapping.dat.gz"

# Input/output
ANCHORED_FILE = Path("./pubtator/output/m2tqa_tiered_anchored_v3.jsonl")
VARIANT_MAPPING = Path("./vep/input/variant_mapping.jsonl")
OUTPUT_FILE = Path("./uniprot/output/variants_with_uniprot.jsonl")
STATS_FILE = Path("./uniprot/output/mapping_stats.json")


def download_mapping_file():
    """Download the UniProt ID mapping file if not already present."""
    if MAPPING_FILE.exists():
        size_mb = MAPPING_FILE.stat().st_size / (1024 * 1024)
        print(f"Using cached mapping file: {MAPPING_FILE} ({size_mb:.1f} MB)")
        return

    print(f"Downloading UniProt ID mapping file...")
    print(f"  URL: {MAPPING_URL}")
    print(f"  Destination: {MAPPING_FILE}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Download with progress
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  Downloaded: {mb:.1f} MB / {total_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)

    urllib.request.urlretrieve(MAPPING_URL, MAPPING_FILE, reporthook)
    print()
    print(f"  Download complete!")


def load_gene_to_uniprot_mapping():
    """
    Load NCBI GeneID to UniProt mapping from the downloaded file.

    File format: UniProtKB-AC<tab>ID_type<tab>ID
    We want lines where ID_type == "GeneID"

    Returns dict: gene_id -> list of UniProt accessions
    """
    print(f"\nLoading GeneID → UniProt mapping...")

    gene_to_uniprot = defaultdict(list)

    with gzip.open(MAPPING_FILE, 'rt') as f:
        for line in tqdm(f, desc="Parsing mapping file"):
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue

            uniprot_acc, id_type, id_value = parts

            if id_type == "GeneID":
                gene_to_uniprot[id_value].append(uniprot_acc)

    print(f"  Loaded {len(gene_to_uniprot):,} gene IDs with UniProt mappings")

    # Count unique UniProt accessions
    all_uniprot = set()
    for accs in gene_to_uniprot.values():
        all_uniprot.update(accs)
    print(f"  Total unique UniProt accessions: {len(all_uniprot):,}")

    return dict(gene_to_uniprot)


def map_variants_to_uniprot(gene_to_uniprot: dict):
    """Map anchored variants to UniProt accessions."""
    print(f"\nMapping variants to UniProt...")
    print(f"  Input: {ANCHORED_FILE}")

    stats = {
        "total_records": 0,
        "human_records": 0,
        "with_gene_id": 0,
        "mapped_to_uniprot": 0,
        "unique_genes_mapped": set(),
        "unique_uniprot_ids": set(),
        "multiple_uniprot_matches": 0,
    }

    mapped_records = []

    with open(ANCHORED_FILE) as f:
        for line in tqdm(f, desc="Mapping variants"):
            record = json.loads(line)
            stats["total_records"] += 1

            # Filter for human only
            taxon = record.get("ncbi_taxon_id")
            if taxon != "9606":
                continue

            stats["human_records"] += 1

            gene_id = record.get("anchor_gene_id")
            if not gene_id:
                continue

            stats["with_gene_id"] += 1

            # Look up UniProt accessions
            uniprot_ids = gene_to_uniprot.get(gene_id, [])

            if uniprot_ids:
                stats["mapped_to_uniprot"] += 1
                stats["unique_genes_mapped"].add(gene_id)
                stats["unique_uniprot_ids"].update(uniprot_ids)

                if len(uniprot_ids) > 1:
                    stats["multiple_uniprot_matches"] += 1

                # Add UniProt info to record
                record["uniprot_ids"] = uniprot_ids
                record["uniprot_primary"] = uniprot_ids[0]  # First is typically reviewed/canonical
                mapped_records.append(record)

    # Convert sets to counts for JSON serialization
    stats["unique_genes_mapped"] = len(stats["unique_genes_mapped"])
    stats["unique_uniprot_ids"] = len(stats["unique_uniprot_ids"])

    return mapped_records, stats


def main():
    print("=" * 60)
    print("NCBI Gene ID → UniProt Mapping")
    print("=" * 60)

    # Step 1: Download mapping file
    download_mapping_file()

    # Step 2: Load GeneID to UniProt mapping
    gene_to_uniprot = load_gene_to_uniprot_mapping()

    # Step 3: Map variants
    mapped_records, stats = map_variants_to_uniprot(gene_to_uniprot)

    # Step 4: Write output
    print(f"\nWriting output: {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        for record in mapped_records:
            f.write(json.dumps(record) + "\n")

    # Step 5: Write stats
    print(f"Writing stats: {STATS_FILE}")
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

    # Summary
    print()
    print("=" * 60)
    print("MAPPING SUMMARY")
    print("=" * 60)
    print(f"Total records: {stats['total_records']:,}")
    print(f"Human records: {stats['human_records']:,}")
    print(f"With gene ID: {stats['with_gene_id']:,}")
    print(f"Mapped to UniProt: {stats['mapped_to_uniprot']:,} ({stats['mapped_to_uniprot']/stats['human_records']*100:.1f}%)")
    print(f"Unique genes mapped: {stats['unique_genes_mapped']:,}")
    print(f"Unique UniProt IDs: {stats['unique_uniprot_ids']:,}")
    print(f"Records with multiple UniProt matches: {stats['multiple_uniprot_matches']:,}")
    print()
    print(f"Output: {OUTPUT_FILE}")
    print(f"Stats: {STATS_FILE}")


if __name__ == "__main__":
    main()
