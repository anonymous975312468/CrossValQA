#!/usr/bin/env python3
"""
Master Grounding Pipeline for M2TQA Dataset

This script performs three levels of gene-to-UniProt grounding:
1. DIRECT: Records with taxon ID + anchor_gene_id from PubTator
2. RECOVERED_GENEID: Records without taxon but gene ID found in species mappings
3. RECOVERED_SYMBOL: Records with recovered_gene but no anchor_gene_id,
   recovered via NCBI gene symbol lookup

Outputs detailed statistics for the ICML paper.
"""

import csv
import gzip
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# =============================================================================
# PATHS
# =============================================================================
ANCHORED_FILE = Path("./pubtator/output/m2tqa_tiered_anchored_v3.jsonl")
NCBI_GENE_INFO = Path("./mappings/Homo_sapiens.gene_info")
UNIPROT_MAPPING_DIR = Path("./uniprot/input")

OUTPUT_FILE = Path("./uniprot/output/m2tqa_grounded_final.jsonl")
STATS_FILE = Path("./uniprot/output/grounding_stats_final.json")

# UniProt FTP
UNIPROT_FTP = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism"

SPECIES_FILES = {
    "9606": "HUMAN_9606_idmapping.dat.gz",
    "10090": "MOUSE_10090_idmapping.dat.gz",
    "10116": "RAT_10116_idmapping.dat.gz",
    "7955": "DANRE_7955_idmapping.dat.gz",
    "7227": "DROME_7227_idmapping.dat.gz",
    "4932": "YEAST_559292_idmapping.dat.gz",
    "559292": "YEAST_559292_idmapping.dat.gz",
    "6239": "CAEEL_6239_idmapping.dat.gz",
    "9031": "CHICK_9031_idmapping.dat.gz",
}


def load_ncbi_symbol_to_geneid():
    """
    Load NCBI Gene Info file to create symbol -> gene_id mapping.
    Handles both official symbols (priority) and aliases/synonyms.
    """
    print("Loading NCBI Gene Symbol -> Gene ID mapping...")
    symbol_to_id = {}
    id_to_symbol = {}  # For getting official symbol later

    with open(NCBI_GENE_INFO, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            gene_id = row['GeneID']
            official_symbol = row['Symbol']
            synonyms = row['Synonyms'].split('|') if row['Synonyms'] != '-' else []

            # Store official symbol -> ID (highest priority)
            symbol_to_id[official_symbol.upper()] = (gene_id, official_symbol)
            id_to_symbol[gene_id] = official_symbol

            # Store synonyms -> ID (only if not already taken by official symbol)
            for syn in synonyms:
                if syn and syn.upper() not in symbol_to_id:
                    symbol_to_id[syn.upper()] = (gene_id, official_symbol)

    print(f"  Loaded {len(symbol_to_id):,} symbols/aliases -> {len(id_to_symbol):,} genes")
    return symbol_to_id, id_to_symbol


def load_uniprot_mappings():
    """Load Gene ID -> UniProt mappings for all available species."""
    print("\nLoading UniProt ID mappings...")

    # gene_id -> list of UniProt accessions (per species)
    species_gene_to_uniprot = {}

    # Combined lookup for gene ID recovery (human priority)
    all_gene_to_uniprot = {}

    for taxon, filename in SPECIES_FILES.items():
        filepath = UNIPROT_MAPPING_DIR / filename
        if not filepath.exists():
            print(f"  {taxon}: File not found, skipping")
            continue

        gene_map = defaultdict(list)
        with gzip.open(filepath, 'rt') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3 and parts[1] == "GeneID":
                    gene_map[parts[2]].append(parts[0])

        species_gene_to_uniprot[taxon] = dict(gene_map)
        print(f"  {taxon}: {len(gene_map):,} gene IDs")

        # Add to combined lookup (human gets priority)
        if taxon == "9606":
            for gid, uniprots in gene_map.items():
                all_gene_to_uniprot[gid] = ("9606", uniprots)
        else:
            for gid, uniprots in gene_map.items():
                if gid not in all_gene_to_uniprot:
                    all_gene_to_uniprot[gid] = (taxon, uniprots)

    # Yeast alias
    if "559292" in species_gene_to_uniprot:
        species_gene_to_uniprot["4932"] = species_gene_to_uniprot["559292"]

    return species_gene_to_uniprot, all_gene_to_uniprot


def main():
    print("=" * 70)
    print("M2TQA MASTER GROUNDING PIPELINE")
    print("=" * 70)

    # Load all mappings
    symbol_to_geneid, id_to_symbol = load_ncbi_symbol_to_geneid()
    species_gene_to_uniprot, all_gene_to_uniprot = load_uniprot_mappings()

    # Human-specific UniProt mapping for symbol recovery
    human_gene_to_uniprot = species_gene_to_uniprot.get("9606", {})

    # Statistics
    stats = {
        "total_records": 0,
        "grounded": {
            "direct": 0,
            "recovered_geneid": 0,
            "recovered_symbol": 0,
            "total": 0,
        },
        "not_grounded": {
            "no_gene_info": 0,
            "unsupported_species": 0,
            "gene_not_in_uniprot": 0,
            "symbol_not_found": 0,
            "symbol_no_uniprot": 0,
        },
        "by_species": defaultdict(lambda: {"total": 0, "grounded": 0}),
        "symbol_recovery_details": {
            "attempted": 0,
            "symbol_found": 0,
            "uniprot_found": 0,
        },
    }

    grounded_records = []

    print("\nProcessing records...")
    with open(ANCHORED_FILE) as f:
        for line in tqdm(f, desc="Grounding"):
            record = json.loads(line)
            stats["total_records"] += 1

            taxon = record.get("ncbi_taxon_id")
            anchor_gene_id = record.get("anchor_gene_id")
            recovered_gene = record.get("recovered_gene")

            grounded = False
            grounding_method = None

            # =========================================================
            # Strategy 1: DIRECT - Has taxon + anchor_gene_id
            # =========================================================
            if taxon and anchor_gene_id:
                stats["by_species"][taxon]["total"] += 1

                if taxon in species_gene_to_uniprot:
                    uniprot_ids = species_gene_to_uniprot[taxon].get(anchor_gene_id, [])
                    if uniprot_ids:
                        record["uniprot_ids"] = uniprot_ids
                        record["uniprot_primary"] = uniprot_ids[0]
                        record["grounding_method"] = "direct"
                        stats["grounded"]["direct"] += 1
                        stats["by_species"][taxon]["grounded"] += 1
                        grounded = True
                    else:
                        stats["not_grounded"]["gene_not_in_uniprot"] += 1
                else:
                    stats["not_grounded"]["unsupported_species"] += 1

            # =========================================================
            # Strategy 2: RECOVERED_GENEID - No taxon but has anchor_gene_id
            # =========================================================
            elif anchor_gene_id and not taxon:
                if anchor_gene_id in all_gene_to_uniprot:
                    inferred_taxon, uniprot_ids = all_gene_to_uniprot[anchor_gene_id]
                    record["ncbi_taxon_id"] = inferred_taxon
                    record["uniprot_ids"] = uniprot_ids
                    record["uniprot_primary"] = uniprot_ids[0]
                    record["grounding_method"] = "recovered_geneid"
                    record["taxon_inferred"] = True
                    stats["grounded"]["recovered_geneid"] += 1
                    grounded = True
                else:
                    stats["not_grounded"]["gene_not_in_uniprot"] += 1

            # =========================================================
            # Strategy 3: RECOVERED_SYMBOL - Has recovered_gene but no anchor_gene_id
            # =========================================================
            elif recovered_gene and not anchor_gene_id:
                stats["symbol_recovery_details"]["attempted"] += 1

                # Look up symbol in NCBI gene info
                symbol_upper = recovered_gene.upper()
                if symbol_upper in symbol_to_geneid:
                    gene_id, official_symbol = symbol_to_geneid[symbol_upper]
                    stats["symbol_recovery_details"]["symbol_found"] += 1

                    # Now look up in human UniProt mapping
                    uniprot_ids = human_gene_to_uniprot.get(gene_id, [])
                    if uniprot_ids:
                        record["anchor_gene_id"] = gene_id
                        record["anchor_gene_name"] = official_symbol
                        record["ncbi_taxon_id"] = "9606"
                        record["uniprot_ids"] = uniprot_ids
                        record["uniprot_primary"] = uniprot_ids[0]
                        record["grounding_method"] = "recovered_symbol"
                        record["symbol_matched"] = recovered_gene
                        record["taxon_inferred"] = True
                        stats["grounded"]["recovered_symbol"] += 1
                        stats["symbol_recovery_details"]["uniprot_found"] += 1
                        grounded = True
                    else:
                        stats["not_grounded"]["symbol_no_uniprot"] += 1
                else:
                    stats["not_grounded"]["symbol_not_found"] += 1

            # =========================================================
            # No grounding possible
            # =========================================================
            else:
                stats["not_grounded"]["no_gene_info"] += 1

            if grounded:
                grounded_records.append(record)

    # Calculate totals
    stats["grounded"]["total"] = (
        stats["grounded"]["direct"] +
        stats["grounded"]["recovered_geneid"] +
        stats["grounded"]["recovered_symbol"]
    )
    stats["by_species"] = dict(stats["by_species"])

    # Write output
    print(f"\nWriting grounded records: {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        for record in grounded_records:
            f.write(json.dumps(record) + "\n")

    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    total = stats["total_records"]
    grounded_total = stats["grounded"]["total"]

    print()
    print("=" * 70)
    print("GROUNDING RESULTS FOR ICML PAPER")
    print("=" * 70)
    print()
    print(f"Total M2TQA Records:              {total:>10,}")
    print()
    print("GROUNDED RECORDS:")
    print(f"  Direct (PubTator annotation):   {stats['grounded']['direct']:>10,}  ({stats['grounded']['direct']/total*100:.1f}%)")
    print(f"  Recovered via Gene ID:          {stats['grounded']['recovered_geneid']:>10,}  ({stats['grounded']['recovered_geneid']/total*100:.1f}%)")
    print(f"  Recovered via Symbol Lookup:    {stats['grounded']['recovered_symbol']:>10,}  ({stats['grounded']['recovered_symbol']/total*100:.1f}%)")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  TOTAL GROUNDED:                 {grounded_total:>10,}  ({grounded_total/total*100:.1f}%)")
    print()
    print("NOT GROUNDED:")
    for reason, count in stats["not_grounded"].items():
        print(f"  {reason}: {count:,}")
    print()
    print("SYMBOL RECOVERY DETAILS:")
    sr = stats["symbol_recovery_details"]
    print(f"  Attempted:     {sr['attempted']:,}")
    print(f"  Symbol found:  {sr['symbol_found']:,} ({sr['symbol_found']/sr['attempted']*100:.1f}% of attempted)")
    print(f"  UniProt found: {sr['uniprot_found']:,} ({sr['uniprot_found']/sr['attempted']*100:.1f}% of attempted)")
    print()
    print(f"Output: {OUTPUT_FILE}")
    print(f"Stats:  {STATS_FILE}")


if __name__ == "__main__":
    main()
