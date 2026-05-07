#!/usr/bin/env python3
"""
Batch Mutation Extractor - Parallelized

Processes all articles in the database, extracts mutations using regex.py,
and overwrites the 'proteins' field with extracted mutations.

Usage:
    python batch_extract_mutations.py [--workers N] [--output PATH]
"""

import json
import sys
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import time
from pathlib import Path

# Add parent directory for config, then phase_0 for sibling imports (regex.py)
_phase0_dir = str(Path(__file__).resolve().parent)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, _phase0_dir)  # must be first to shadow the 'regex' pip package
from config import ARTICLES_FILE, DATABASE_DIR, EXTRACTION_STATS_FILE
from regex import extract_all_variants, get_mutations_for_llm

def process_single_article(args):
    """
    Process a single article and extract mutations.
    
    Args:
        args: tuple of (pmid, article_text)
    
    Returns:
        tuple of (pmid, mutations_list, mutation_count)
    """
    pmid, article_text = args
    
    try:
        if not article_text or not article_text.strip():
            return (pmid, [], 0, "empty_article")
        
        # Run extraction
        results = extract_all_variants(article_text, apply_section_filter=True)
        llm_data = get_mutations_for_llm(results)
        
        mutations = llm_data["mutations"]
        count = len(mutations)
        
        return (pmid, mutations, count, None)
    
    except Exception as e:
        return (pmid, [], 0, str(e))


def main():
    parser = argparse.ArgumentParser(description="Batch extract mutations from articles")
    parser.add_argument(
        "--input",
        default=str(ARTICLES_FILE),
        help="Input JSON file path"
    )
    parser.add_argument(
        "--output",
        default=str(DATABASE_DIR / "pubmed_to_proteins_with_articles.mutations_extracted.json"),
        help="Output JSON file path"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help=f"Number of parallel workers (default: {os.cpu_count()})"
    )
    parser.add_argument(
        "--stats-output",
        default=str(EXTRACTION_STATS_FILE),
        help="Output path for extraction statistics"
    )
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    start_time = time.time()
    
    with open(args.input, "r") as f:
        data = json.load(f)
    
    total_entries = len(data)
    print(f"Loaded {total_entries:,} entries in {time.time() - start_time:.1f}s")
    print(f"Using {args.workers} workers")
    
    # Prepare work items
    work_items = [(pmid, entry.get("article", "")) for pmid, entry in data.items()]
    
    # Track statistics
    mutation_counts = Counter()
    zero_mutation_pmids = []
    error_pmids = []
    total_mutations_extracted = 0
    
    # Process in parallel
    print(f"\nProcessing articles...")
    processed = 0
    start_process_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_article, item): item[0] for item in work_items}
        
        for future in as_completed(futures):
            pmid, mutations, count, error = future.result()
            
            # Update the data
            data[pmid]["proteins"] = mutations
            
            # Track stats
            mutation_counts[count] += 1
            total_mutations_extracted += count
            
            if count == 0:
                zero_mutation_pmids.append({"pmid": pmid, "error": error})
            
            if error:
                error_pmids.append({"pmid": pmid, "error": error})
            
            processed += 1
            
            # Progress update every 1000 articles
            if processed % 1000 == 0:
                elapsed = time.time() - start_process_time
                rate = processed / elapsed
                eta = (total_entries - processed) / rate
                print(f"  Processed {processed:,}/{total_entries:,} ({processed/total_entries*100:.1f}%) "
                      f"- {rate:.0f} articles/sec - ETA: {eta/60:.1f} min")
    
    total_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total articles processed: {total_entries:,}")
    print(f"Total mutations extracted: {total_mutations_extracted:,}")
    print(f"Average mutations per article: {total_mutations_extracted/total_entries:.2f}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Processing rate: {total_entries/total_time:.0f} articles/sec")
    
    print(f"\n--- Mutation Count Distribution ---")
    for count in sorted(mutation_counts.keys())[:20]:
        num_articles = mutation_counts[count]
        pct = num_articles / total_entries * 100
        bar = "█" * int(pct / 2)
        print(f"  {count:3d} mutations: {num_articles:6,} articles ({pct:5.1f}%) {bar}")
    
    if len(mutation_counts) > 20:
        remaining = sum(v for k, v in mutation_counts.items() if k >= 20)
        print(f"  20+ mutations: {remaining:6,} articles")
    
    print(f"\nArticles with 0 mutations: {mutation_counts[0]:,} ({mutation_counts[0]/total_entries*100:.1f}%)")
    print(f"Articles with errors: {len(error_pmids):,}")
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(data, f)
    print(f"Saved.")
    
    # Save statistics
    stats = {
        "total_articles": total_entries,
        "total_mutations_extracted": total_mutations_extracted,
        "avg_mutations_per_article": total_mutations_extracted / total_entries,
        "processing_time_seconds": total_time,
        "articles_per_second": total_entries / total_time,
        "mutation_count_distribution": dict(mutation_counts),
        "zero_mutation_pmids": zero_mutation_pmids[:1000],  # Cap at 1000 to avoid huge file
        "zero_mutation_count": len(zero_mutation_pmids),
        "error_pmids": error_pmids,
    }
    
    print(f"Saving statistics to {args.stats_output}...")
    with open(args.stats_output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved.")
    
    # Print sample of zero-mutation PMIDs for debugging
    if zero_mutation_pmids:
        print(f"\n--- Sample PMIDs with 0 mutations (first 10) ---")
        for item in zero_mutation_pmids[:10]:
            pmid = item["pmid"]
            error = item["error"]
            article_len = len(data[pmid].get("article", ""))
            error_str = f" (error: {error})" if error else ""
            print(f"  PMID {pmid}: article length {article_len:,}{error_str}")


if __name__ == "__main__":
    main()