#!/usr/bin/env python3
"""
LLM Mutation Verification Pipeline

Verifies that mutations are functionally discussed in papers using LLM via OpenRouter.
Uses prefiltering to reduce costs and parallelization for speed.

Usage:
    python verify_mutations.py --model google/gemini-2.5-flash-lite --workers 50
    
Resume interrupted run:
    python verify_mutations.py --model google/gemini-2.5-flash-lite --workers 50 --resume
"""

import json
import os
import sys
import argparse
import time
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path

# Add parent directory for config, then phase_0 for sibling imports (prefilter.py)
_phase0_dir = str(Path(__file__).resolve().parent)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, _phase0_dir)  # must be first for sibling module imports
from config import (
    ARTICLES_FILE, VERIFIED_MUTATIONS_FILE, VERIFICATION_STATS_FILE,
    FAILED_BATCHES_FILE, DATABASE_DIR,
)
from prefilter import prefilter_mutations, split_into_sentences

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PATHS FOR CHECKPOINTS AND FAILED BATCHES
# ============================================================================
FAILED_BATCHES_PATH = str(FAILED_BATCHES_FILE)
CHECKPOINT_DIR = str(DATABASE_DIR / "checkpoints")


@dataclass
class VerifiedMutation:
    """Result of LLM verification for a single mutation."""
    mutation: str
    is_discussed: bool
    consequence_summary: Optional[str]
    evidence_sentences: List[int]
    confidence: str  # high, medium, low
    gene_match: Optional[bool] = None  # None = no target gene (original pipeline)


@dataclass 
class ArticleResult:
    """Result of processing a single article."""
    pmid: str
    verified_mutations: List[VerifiedMutation]
    filtered_gene_names: List[str]
    just_mentioned: List[str]
    error: Optional[str] = None


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

SYSTEM_PROMPT = """You are a scientific literature analyst specializing in genetics and molecular biology. 
Your task is to determine whether genetic mutations are functionally discussed in research papers.

A mutation is "functionally discussed" if the paper describes:
- Its effect on protein function, stability, or expression
- Its clinical/phenotypic consequences
- Experimental results demonstrating its impact
- Its pathogenicity classification with supporting evidence

A mutation is NOT functionally discussed if it is:
- Simply listed in a table without discussion
- Mentioned as "previously reported" without new findings
- Referenced only as part of methodology
- Mentioned in passing without functional context

CRITICAL: Your consequence_summary must ONLY include information that is DIRECTLY stated in the cited sentences. 
Do not make inferences or logical deductions. If a sentence says a mutation "was found in a patient with disease X", 
do not infer it is pathogenic unless the word "pathogenic" (or equivalent) explicitly appears.
Every claim in your consequence_summary must be directly verifiable in the cited evidence sentences."""

USER_PROMPT_TEMPLATE = """Analyze the following mutations from a scientific paper and determine which ones are functionally discussed.

MUTATIONS TO ANALYZE:
{mutations_list}

RELEVANT PAPER EXCERPTS (numbered sentences):
{numbered_sentences}

For each mutation, respond in this exact JSON format:
{{
  "results": [
    {{
      "mutation": "<mutation_id>",
      "is_discussed": true/false,
      "consequence_summary": "<brief summary using ONLY information directly stated in cited sentences - no inferences>",
      "evidence_sentences": [<list of sentence numbers that DIRECTLY support your summary>],
      "confidence": "high/medium/low"
    }}
  ]
}}

IMPORTANT RULES:
1. Only include mutations from the provided list
2. Every claim in consequence_summary must be DIRECTLY stated in at least one cited sentence
3. Do not infer pathogenicity - only state it if explicitly mentioned in the text
4. Do not extrapolate from related mutations - each mutation must have its own direct evidence
5. If a mutation only has indirect or inferred evidence, mark is_discussed as false
6. Be strict - when in doubt, mark as not discussed"""

# Gene-aware variant for expansion pipeline
USER_PROMPT_TEMPLATE_GENE = """Analyze the following mutations from a scientific paper. Determine which ones are functionally discussed AND associated with the target gene.

TARGET GENE: {target_gene}

MUTATIONS TO ANALYZE:
{mutations_list}

RELEVANT PAPER EXCERPTS (numbered sentences):
{numbered_sentences}

For each mutation, respond in this exact JSON format:
{{
  "results": [
    {{
      "mutation": "<mutation_id>",
      "is_discussed": true/false,
      "gene_match": true/false,
      "consequence_summary": "<brief summary using ONLY information directly stated in cited sentences - no inferences>",
      "evidence_sentences": [<list of sentence numbers that DIRECTLY support your summary>],
      "confidence": "high/medium/low"
    }}
  ]
}}

IMPORTANT RULES:
1. Only include mutations from the provided list
2. Every claim in consequence_summary must be DIRECTLY stated in at least one cited sentence
3. Do not infer pathogenicity - only state it if explicitly mentioned in the text
4. Do not extrapolate from related mutations - each mutation must have its own direct evidence
5. If a mutation only has indirect or inferred evidence, mark is_discussed as false
6. Be strict - when in doubt, mark as not discussed
7. gene_match: Is this mutation associated with {target_gene}? Mark true only if the text explicitly links the mutation to {target_gene} (directly names it, or the surrounding context makes it unambiguous). Mark false if it belongs to a different gene or if the gene association is unclear."""


# ============================================================================
# FAILED BATCH LOGGING
# ============================================================================

def log_failed_batch(pmid: str, mutations: List[str], error: str):
    """Append failed batch to log file for later retry."""
    Path(FAILED_BATCHES_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(FAILED_BATCHES_PATH, "a") as f:
        f.write(json.dumps({
            "pmid": pmid,
            "mutations": mutations,
            "error": error,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }) + "\n")


def get_failed_batch_count() -> int:
    """Count number of failed batches logged."""
    if not Path(FAILED_BATCHES_PATH).exists():
        return 0
    with open(FAILED_BATCHES_PATH, "r") as f:
        return sum(1 for _ in f)


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(results: Dict[str, ArticleResult], checkpoint_id: int, output_path: str):
    """Save checkpoint to disk."""
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(CHECKPOINT_DIR) / f"checkpoint_{checkpoint_id}.json"
    
    output = {}
    for pmid, result in results.items():
        output[pmid] = {
            "verified_mutations": [
                asdict(v) for v in result.verified_mutations if v.is_discussed
            ],
            "unverified_mutations": [
                asdict(v) for v in result.verified_mutations if not v.is_discussed
            ],
            "filtered_gene_names": result.filtered_gene_names,
            "just_mentioned": result.just_mentioned,
            "error": result.error,
        }
    
    with open(checkpoint_path, "w") as f:
        json.dump(output, f)
    
    logger.info(f"Checkpoint saved: {checkpoint_path} ({len(results):,} articles)")


def load_latest_checkpoint() -> Tuple[Dict, int]:
    """Load the latest checkpoint if available."""
    checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        return {}, 0
    
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.json"))
    if not checkpoints:
        return {}, 0
    
    latest = checkpoints[-1]
    checkpoint_id = int(latest.stem.split("_")[1])
    
    logger.info(f"Loading checkpoint: {latest}")
    with open(latest, "r") as f:
        data = json.load(f)
    
    return data, checkpoint_id


def load_existing_results(output_path: str) -> set:
    """Load PMIDs that have already been processed."""
    if not Path(output_path).exists():
        return set()
    
    try:
        with open(output_path, "r") as f:
            data = json.load(f)
        return set(data.keys())
    except (json.JSONDecodeError, IOError):
        return set()


# ============================================================================
# OPENROUTER API
# ============================================================================

class OpenRouterClient:
    """Async client for OpenRouter API."""
    
    def __init__(self, api_key: str, model: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.semaphore = None  # Set during processing
        
    async def complete(
        self, 
        session: aiohttp.ClientSession,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        retries: int = 3,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Make a completion request to OpenRouter with retry logic.
        
        Returns:
            Tuple of (response_text, error_message)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mutation-verification",
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        last_error = None
        for attempt in range(retries):
            async with self.semaphore:
                try:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data["choices"][0]["message"]["content"], None
                        elif response.status == 429:
                            # Rate limited - wait and retry
                            wait_time = 5 * (attempt + 1)
                            logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{retries})")
                            await asyncio.sleep(wait_time)
                            last_error = "rate_limited"
                        else:
                            error_text = await response.text()
                            last_error = f"HTTP {response.status}: {error_text[:200]}"
                            if response.status >= 500:
                                # Server error - retry
                                await asyncio.sleep(2 * (attempt + 1))
                            else:
                                # Client error - don't retry
                                return None, last_error
                except asyncio.TimeoutError:
                    last_error = "timeout"
                    logger.warning(f"Timeout (attempt {attempt + 1}/{retries})")
                    await asyncio.sleep(2)
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Request error: {e} (attempt {attempt + 1}/{retries})")
                    await asyncio.sleep(2)
        
        return None, last_error


# ============================================================================
# MUTATION VERIFICATION
# ============================================================================

def build_numbered_sentences(sentences: List[str], relevant_indices: List[int], context_window: int = 2) -> str:
    """
    Build numbered sentences string for the prompt.
    Includes relevant sentences plus context.
    """
    # Expand indices to include context
    expanded = set()
    for idx in relevant_indices:
        for i in range(max(0, idx - context_window), min(len(sentences), idx + context_window + 1)):
            expanded.add(i)
    
    # Limit total sentences to avoid context overflow
    sorted_indices = sorted(expanded)[:100]
    
    lines = []
    for idx in sorted_indices:
        lines.append(f"[{idx}] {sentences[idx]}")
    
    return "\n".join(lines)


def parse_llm_response(response_text: str, expected_mutations: List[str], pmid: str = None) -> List[VerifiedMutation]:
    """Parse LLM JSON response into VerifiedMutation objects."""
    results = []
    
    try:
        # Try to extract JSON from response
        # Handle case where LLM includes markdown code blocks
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
        else:
            json_str = response_text
        
        data = json.loads(json_str.strip())
        
        for item in data.get("results", []):
            mutation = item.get("mutation", "")
            if mutation in expected_mutations:
                results.append(VerifiedMutation(
                    mutation=mutation,
                    is_discussed=item.get("is_discussed", False),
                    consequence_summary=item.get("consequence_summary"),
                    evidence_sentences=item.get("evidence_sentences", []),
                    confidence=item.get("confidence", "low"),
                    gene_match=item.get("gene_match"),
                ))
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning(f"Failed to parse LLM response: {e}")
        # Log for retry later
        if pmid:
            log_failed_batch(pmid, expected_mutations, f"parse_error: {e}")
    
    return results


async def verify_mutation_batch(
    client: OpenRouterClient,
    session: aiohttp.ClientSession,
    mutations: List[Dict],
    sentences: List[str],
    pmid: str = None,
    target_gene: Optional[str] = None,
) -> List[VerifiedMutation]:
    """Verify a batch of mutations using LLM.

    If target_gene is provided, the prompt also asks the LLM to determine
    whether each mutation is associated with that gene (expansion mode).
    """

    # Build mutations list for prompt
    mutations_list = "\n".join([f"- {m['mutation']}" for m in mutations])

    # Get all relevant sentence indices
    relevant_indices = set()
    for m in mutations:
        relevant_indices.update(m.get("mutation_sentences", []))
        relevant_indices.update(m.get("evidence_sentences", []))

    # Build numbered sentences
    numbered_sentences = build_numbered_sentences(sentences, list(relevant_indices))

    # Build prompt — use gene-aware template if target_gene is set
    if target_gene:
        user_prompt = USER_PROMPT_TEMPLATE_GENE.format(
            target_gene=target_gene,
            mutations_list=mutations_list,
            numbered_sentences=numbered_sentences
        )
    else:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            mutations_list=mutations_list,
            numbered_sentences=numbered_sentences
        )

    # Call LLM
    response, error = await client.complete(session, SYSTEM_PROMPT, user_prompt)

    if error:
        logger.warning(f"LLM error for {pmid}: {error}")
        mutation_names = [m["mutation"] for m in mutations]
        log_failed_batch(pmid, mutation_names, f"llm_error: {error}")
        return []

    # Parse response
    expected = [m["mutation"] for m in mutations]
    return parse_llm_response(response, expected, pmid)


async def process_article(
    client: OpenRouterClient,
    session: aiohttp.ClientSession,
    pmid: str,
    article_text: str,
    mutations: List[str],
    batch_size: int = 10,
    target_gene: Optional[str] = None,
) -> ArticleResult:
    """Process a single article - prefilter and verify mutations.

    If target_gene is set (expansion mode), the LLM also checks gene association
    and only mutations with gene_match=True are kept as verified.
    """

    try:
        # Prefilter mutations
        prefilter_result = prefilter_mutations(article_text, mutations)

        likely_discussed = prefilter_result["likely_discussed"]
        just_mentioned = prefilter_result["likely_just_mentioned"]
        filtered_genes = prefilter_result["filtered_gene_names"]
        sentences = prefilter_result["sentences"]

        if not likely_discussed:
            return ArticleResult(
                pmid=pmid,
                verified_mutations=[],
                filtered_gene_names=filtered_genes,
                just_mentioned=just_mentioned,
            )

        # Batch mutations for LLM verification
        verified = []
        for i in range(0, len(likely_discussed), batch_size):
            batch = likely_discussed[i:i + batch_size]
            batch_results = await verify_mutation_batch(
                client, session, batch, sentences, pmid, target_gene
            )
            verified.extend(batch_results)

        return ArticleResult(
            pmid=pmid,
            verified_mutations=verified,
            filtered_gene_names=filtered_genes,
            just_mentioned=just_mentioned,
        )

    except Exception as e:
        logger.error(f"Error processing {pmid}: {e}")
        log_failed_batch(pmid, mutations, f"processing_error: {e}")
        return ArticleResult(
            pmid=pmid,
            verified_mutations=[],
            filtered_gene_names=[],
            just_mentioned=[],
            error=str(e)
        )


# ============================================================================
# BATCH PROCESSING
# ============================================================================

# Replace the process_all_articles function with this chunked version:

async def process_all_articles(
    data: Dict[str, Dict],
    client: OpenRouterClient,
    output_path: str,
    max_concurrent: int = 50,
    batch_size: int = 10,
    checkpoint_interval: int = 1000,
    skip_pmids: set = None,
    chunk_size: int = 2000,
) -> Dict[str, ArticleResult]:
    """Process all articles with parallel LLM calls and checkpointing.

    Each entry in data can optionally include a 'gene' key. When present,
    the LLM verification will also check gene association (expansion mode).
    """

    client.semaphore = asyncio.Semaphore(max_concurrent)
    skip_pmids = skip_pmids or set()

    # Filter out already-processed PMIDs
    articles_to_process = {k: v for k, v in data.items() if k not in skip_pmids}

    if skip_pmids:
        logger.info(f"Skipping {len(skip_pmids):,} already-processed articles")
    logger.info(f"Processing {len(articles_to_process):,} articles in chunks of {chunk_size}")

    results = {}
    total = len(articles_to_process)
    processed = 0
    start_time = time.time()
    checkpoint_counter = 0

    # Convert to list for chunking
    article_items = list(articles_to_process.items())

    async with aiohttp.ClientSession() as session:
        # Process in chunks to avoid memory issues
        for chunk_start in range(0, len(article_items), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(article_items))
            chunk = article_items[chunk_start:chunk_end]

            logger.info(f"Processing chunk {chunk_start//chunk_size + 1} ({chunk_start+1}-{chunk_end} of {total})")

            # Create tasks for this chunk only
            tasks = {}
            for pmid, entry in chunk:
                task = asyncio.create_task(
                    process_article(
                        client, session, pmid,
                        entry.get("article", ""),
                        entry.get("proteins", []),
                        batch_size,
                        target_gene=entry.get("gene"),
                    )
                )
                tasks[task] = pmid
            
            # Process this chunk as they complete
            for coro in asyncio.as_completed(tasks.keys()):
                try:
                    result = await coro
                    results[result.pmid] = result
                except Exception as e:
                    pmid = tasks.get(coro, "unknown")
                    logger.error(f"Unexpected error for {pmid}: {e}")
                    log_failed_batch(pmid, [], f"unexpected_error: {e}")
                    results[pmid] = ArticleResult(
                        pmid=pmid,
                        verified_mutations=[],
                        filtered_gene_names=[],
                        just_mentioned=[],
                        error=str(e)
                    )
                
                processed += 1
                checkpoint_counter += 1
                
                # Checkpoint save
                if checkpoint_counter >= checkpoint_interval:
                    save_checkpoint(results, processed, output_path)
                    checkpoint_counter = 0
                
                if processed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    eta = (total - processed) / rate if rate > 0 else 0
                    
                    total_verified = sum(
                        len([v for v in r.verified_mutations if v.is_discussed])
                        for r in results.values()
                    )
                    
                    failed_count = get_failed_batch_count()
                    
                    logger.info(
                        f"Processed {processed:,}/{total:,} ({processed/total*100:.1f}%) "
                        f"- {rate:.1f}/sec - ETA: {eta/60:.1f}min "
                        f"- Verified: {total_verified:,} - Failed batches: {failed_count}"
                    )
    
    return results

def save_results(results: Dict[str, ArticleResult], output_path: str):
    """Save results to JSON file."""
    
    output = {}
    for pmid, result in results.items():
        output[pmid] = {
            "verified_mutations": [
                asdict(v) for v in result.verified_mutations if v.is_discussed
            ],
            "unverified_mutations": [
                asdict(v) for v in result.verified_mutations if not v.is_discussed
            ],
            "filtered_gene_names": result.filtered_gene_names,
            "just_mentioned": result.just_mentioned,
            "error": result.error,
        }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")


def merge_with_existing(new_results: Dict[str, ArticleResult], output_path: str) -> Dict[str, ArticleResult]:
    """Merge new results with existing results file."""
    if not Path(output_path).exists():
        return new_results
    
    try:
        with open(output_path, "r") as f:
            existing = json.load(f)
        
        # Convert existing to ArticleResult objects
        merged = {}
        for pmid, data in existing.items():
            merged[pmid] = ArticleResult(
                pmid=pmid,
                verified_mutations=[
                    VerifiedMutation(**v) for v in data.get("verified_mutations", [])
                ] + [
                    VerifiedMutation(**v) for v in data.get("unverified_mutations", [])
                ],
                filtered_gene_names=data.get("filtered_gene_names", []),
                just_mentioned=data.get("just_mentioned", []),
                error=data.get("error"),
            )
        
        # Add new results (overwrite if exists)
        for pmid, result in new_results.items():
            merged[pmid] = result
        
        return merged
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not load existing results: {e}")
        return new_results


def generate_stats(results: Dict[str, ArticleResult]) -> Dict:
    """Generate summary statistics."""
    
    total_articles = len(results)
    articles_with_verified = sum(
        1 for r in results.values() 
        if any(v.is_discussed for v in r.verified_mutations)
    )
    total_verified = sum(
        len([v for v in r.verified_mutations if v.is_discussed])
        for r in results.values()
    )
    total_unverified = sum(
        len([v for v in r.verified_mutations if not v.is_discussed])
        for r in results.values()
    )
    total_filtered_genes = sum(len(r.filtered_gene_names) for r in results.values())
    total_just_mentioned = sum(len(r.just_mentioned) for r in results.values())
    total_errors = sum(1 for r in results.values() if r.error)
    failed_batches = get_failed_batch_count()
    
    return {
        "total_articles": total_articles,
        "articles_with_verified_mutations": articles_with_verified,
        "total_verified_mutations": total_verified,
        "total_unverified_by_llm": total_unverified,
        "total_filtered_gene_names": total_filtered_genes,
        "total_just_mentioned": total_just_mentioned,
        "total_errors": total_errors,
        "failed_batches": failed_batches,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Verify mutations with LLM")
    parser.add_argument(
        "--input",
        default=str(ARTICLES_FILE),
        help="Input JSON file"
    )
    parser.add_argument(
        "--output",
        default=str(VERIFIED_MUTATIONS_FILE),
        help="Output JSON file"
    )
    parser.add_argument(
        "--stats-output",
        default=str(VERIFICATION_STATS_FILE),
        help="Statistics output file"
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-3-haiku",
        help="OpenRouter model to use"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=50,
        help="Max concurrent API requests"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Mutations per LLM call"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of articles to process (for testing)"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip already-processed PMIDs)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N articles"
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        logger.error("No API key provided. Set OPENROUTER_API_KEY or use --api-key")
        sys.exit(1)
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    with open(args.input, "r") as f:
        data = json.load(f)
    
    if args.limit:
        data = dict(list(data.items())[:args.limit])
        logger.info(f"Limited to {args.limit} articles for testing")
    
    # Check for resume
    skip_pmids = set()
    if args.resume:
        skip_pmids = load_existing_results(args.output)
        if skip_pmids:
            logger.info(f"Resume mode: found {len(skip_pmids):,} already-processed articles")
    
    logger.info(f"Processing {len(data):,} articles")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max concurrent requests: {args.workers}")
    logger.info(f"Checkpoint interval: {args.checkpoint_interval}")
    
    # Clear failed batches log if not resuming
    if not args.resume and Path(FAILED_BATCHES_PATH).exists():
        Path(FAILED_BATCHES_PATH).unlink()
        logger.info("Cleared previous failed batches log")
    
    # Initialize client
    client = OpenRouterClient(args.api_key, args.model)
    
    # Process
    start_time = time.time()
    try:
        results = asyncio.run(
            process_all_articles(
                data, client, args.output,
                args.workers, args.batch_size,
                args.checkpoint_interval, skip_pmids
            )
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user - saving current progress...")
        results = {}  # Will save whatever was checkpointed
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.info("Check checkpoints directory for partial results")
        raise
    
    total_time = time.time() - start_time
    
    # Merge with existing results if resuming
    if args.resume:
        results = merge_with_existing(results, args.output)
    
    # Save results
    save_results(results, args.output)
    
    # Generate and save stats
    stats = generate_stats(results)
    stats["processing_time_seconds"] = total_time
    stats["model"] = args.model
    
    with open(args.stats_output, "w") as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("VERIFICATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total articles: {stats['total_articles']:,}")
    logger.info(f"Articles with verified mutations: {stats['articles_with_verified_mutations']:,}")
    logger.info(f"Total verified mutations: {stats['total_verified_mutations']:,}")
    logger.info(f"Total unverified by LLM: {stats['total_unverified_by_llm']:,}")
    logger.info(f"Total filtered gene names: {stats['total_filtered_gene_names']:,}")
    logger.info(f"Total just mentioned: {stats['total_just_mentioned']:,}")
    logger.info(f"Errors: {stats['total_errors']:,}")
    logger.info(f"Failed batches: {stats['failed_batches']:,}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    
    if stats['failed_batches'] > 0:
        logger.info(f"\nFailed batches logged to: {FAILED_BATCHES_PATH}")
        logger.info("Run with --resume to retry or use retry_failed.py script")


if __name__ == "__main__":
    main()