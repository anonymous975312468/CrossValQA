#!/usr/bin/env python3
"""
Phase 1: Generate QA pairs + rationales from verified mutations.

Features:
- Pre-filters unusable mutations (rsIDs, gene names)
- Checkpoints every N mutations
- Graceful shutdown on Ctrl+C
- Resumes from last checkpoint
- Logs failed mutations for retry

Usage:
    python phase1_generator_qa.py --model google/gemini-2.5-flash-lite --workers 100
    python phase1_generator_qa.py --resume  # Resume from checkpoint
"""

import json
import os
import sys
import argparse
import asyncio
import aiohttp
import re
import time
import logging
import signal
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Set
from datetime import datetime

# Add parent directory for config, and phase_0 for prefilter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase_0"))
from config import (
    VERIFIED_MUTATIONS_FILE, ARTICLES_FILE, PHASE1_OUTPUT_DIR,
    MISMATCHED_PMIDS_FILE as _MISMATCHED_PMIDS_FILE, DEFAULT_PHASE1_MODEL,
)
from openrouter import OpenRouterClient
from prefilter import split_into_sentences, is_garbage_sentence

# ============================================================================
# CONFIG
# ============================================================================

INPUT_VERIFIED = str(VERIFIED_MUTATIONS_FILE)
INPUT_ARTICLES = str(ARTICLES_FILE)
OUTPUT_DIR = str(PHASE1_OUTPUT_DIR)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase1_qa_pairs.jsonl")
ERROR_FILE = os.path.join(OUTPUT_DIR, "phase1_errors.jsonl")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")
FAILED_FILE = os.path.join(OUTPUT_DIR, "failed_mutations.jsonl")

MISMATCHED_PMIDS_FILE = str(_MISMATCHED_PMIDS_FILE)


CHECKPOINT_INTERVAL = 1000  # Save checkpoint every N mutations

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# MUTATION FILTERING
# ============================================================================

def is_usable_mutation(mut: str) -> bool:
    """
    Return True if mutation has sequence information.
    Excludes rsIDs and gene names.
    """
    # rsIDs - NOT usable without database lookup
    if re.match(r'^rs\d+$', mut):
        return False
    
    # Known gene name patterns that slip through
    if re.match(r'^SEC\d+[A-Z]$', mut):  # SEC16A, SEC23A, etc.
        return False
    
    # Everything else is likely usable
    return True


def filter_evidence_sentences(sentences: List[str], evidence_ids: List[int]) -> List[Dict]:
    """Filter and return good evidence sentences."""
    filtered = []
    for eid in evidence_ids:
        try:
            eid_int = int(eid)
        except (ValueError, TypeError):
            continue
        
        if eid_int < len(sentences):
            sent = sentences[eid_int]
            if not is_garbage_sentence(sent):
                filtered.append({"id": eid_int, "text": sent.strip()})
    
    return filtered


# ============================================================================
# PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are a scientific QA generator specializing in genetics and molecular biology.

Your task is to generate high-quality question-answer pairs about genetic mutations based on evidence from research papers.

CRITICAL RULES:
1. ALWAYS include the mutation identifier (e.g., c.125C>T, p.Thr42Met) in BOTH the question AND answer
2. Generate 2-3 diverse questions covering different aspects (pathogenicity, disease association, functional effect, mechanism, clinical features)
3. Each answer must be directly supported by the provided evidence sentences
4. Each rationale must:
   - Quote at least one relevant phrase from the evidence (in quotation marks)
   - Cite the specific sentence number(s) using [N] format
   - Explain HOW the quote supports the answer
5. If the evidence supports a negative finding (e.g., "does NOT cause...", "is NOT associated with..."), include it as one of the questions. Do not force negative framing if unsupported by evidence.
6. Do NOT infer information not explicitly stated in the evidence
7. Do NOT use external knowledge about the gene or mutation

QUESTION TYPES TO CONSIDER:
- Pathogenicity: "Is [mutation] pathogenic?"
- Disease association: "What disease/condition is associated with [mutation]?"
- Functional effect: "What is the functional effect of [mutation]?"
- Mechanism: "How does [mutation] affect protein function?"
- Clinical features: "What clinical features are seen with [mutation]?"
- Inheritance: "What is the inheritance pattern of [mutation]?"
- Frequency/prevalence: "How common is [mutation]?"

OUTPUT FORMAT (JSON only):
{
  "qa_pairs": [
    {
      "question": "...",
      "answer": "...",
      "rationale": "...",
      "evidence_sentence_ids": [list of ints],
      "question_type": "pathogenicity|disease|function|mechanism|clinical|inheritance|frequency|other"
    }
  ]
}"""


USER_PROMPT_TEMPLATE = """Generate 2-3 diverse QA pairs for this mutation:

MUTATION: {mutation}

CORE FINDING (use as basis for answers):
{consequence_summary}

EVIDENCE SENTENCES FROM THE PAPER:
{evidence_text}

Remember:
- Include "{mutation}" in every question AND answer
- Quote specific phrases from evidence in rationales
- Cite sentence numbers
- Explain how evidence supports the answer
- Try to cover different question types
- Include negative framing if appropriate

Output JSON only."""



# ============================================================================
# QA GENERATION
# ============================================================================

def parse_llm_response(response: str, mutation: str) -> Optional[List[Dict]]:
    """Parse LLM JSON response into QA pairs."""
    try:
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]
        else:
            json_str = response
        
        data = json.loads(json_str.strip())
        qa_pairs = data.get("qa_pairs", [])
        
        valid_pairs = []
        mutation_lower = mutation.lower()
        for qa in qa_pairs:
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            
            if mutation_lower in question.lower() and mutation_lower in answer.lower():
                valid_pairs.append(qa)
        
        return valid_pairs if valid_pairs else None
        
    except (json.JSONDecodeError, KeyError) as e:
        logger.debug(f"Failed to parse response: {e}")
        return None


async def generate_qa_for_mutation(
    client: OpenRouterClient,
    session: aiohttp.ClientSession,
    pmid: str,
    mutation: str,
    consequence_summary: str,
    evidence_sentences: List[Dict],
) -> Optional[Dict]:
    """Generate QA pairs for a single mutation."""
    
    evidence_text = "\n".join([
        f"[{e['id']}] {e['text']}"
        for e in evidence_sentences
    ])
    
    if not evidence_text.strip():
        return None
    
    user_prompt = USER_PROMPT_TEMPLATE.format(
        mutation=mutation,
        consequence_summary=consequence_summary,
        evidence_text=evidence_text
    )
    
    response, error = await client.complete(session, SYSTEM_PROMPT, user_prompt, max_tokens=1500)
    
    if error:
        return None
    
    qa_pairs = parse_llm_response(response, mutation)
    
    if not qa_pairs:
        return None
    
    return {
        "pmid": pmid,
        "mutation": mutation,
        "consequence_summary": consequence_summary,
        "evidence_sentences": evidence_sentences,
        "qa_pairs": qa_pairs
    }


# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(processed_keys: Set[str], stats: dict):
    """Save checkpoint to disk."""
    checkpoint = {
        "processed_keys": list(processed_keys),
        "stats": stats,
        "timestamp": datetime.now().isoformat()
    }
    
    # Write to temp file first, then rename (atomic)
    temp_file = CHECKPOINT_FILE + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(checkpoint, f)
    os.rename(temp_file, CHECKPOINT_FILE)
    
    logger.info(f"Checkpoint saved: {len(processed_keys):,} mutations processed")


def load_checkpoint() -> tuple[Set[str], dict]:
    """Load checkpoint from disk."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set(), {}
    
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint = json.load(f)
        
        processed_keys = set(checkpoint.get("processed_keys", []))
        stats = checkpoint.get("stats", {})
        
        logger.info(f"Loaded checkpoint: {len(processed_keys):,} mutations already processed")
        return processed_keys, stats
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return set(), {}


def load_processed_from_output() -> Set[str]:
    """Load already processed keys from output file."""
    processed = set()
    if not os.path.exists(OUTPUT_FILE):
        return processed
    
    with open(OUTPUT_FILE, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                key = f"{obj['pmid']}:{obj['mutation']}"
                processed.add(key)
            except:
                continue
    
    return processed


# ============================================================================
# BATCH PROCESSING
# ============================================================================

@dataclass
class ProcessingStats:
    total_mutations: int = 0
    processed: int = 0
    successful: int = 0
    failed: int = 0
    total_qa_pairs: int = 0
    skipped_rsid: int = 0
    skipped_short_consequence: int = 0
    skipped_no_evidence: int = 0
    skipped_already_processed: int = 0
    skipped_mismatched_pmids: int = 0


def append_jsonl(path: str, obj: Dict):
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    logger.info("Shutdown requested, finishing current batch...")
    shutdown_requested = True


async def process_batch(
    client: OpenRouterClient,
    session: aiohttp.ClientSession,
    batch: List[tuple],
    stats: ProcessingStats,
    processed_keys: Set[str]
):
    """Process a batch of mutations concurrently."""
    
    tasks = []
    task_info = []
    for pmid, mutation, consequence, evidence in batch:
        task = asyncio.create_task(
            generate_qa_for_mutation(
                client, session, pmid, mutation, consequence, evidence
            )
        )
        tasks.append(task)
        task_info.append((pmid, mutation))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for (pmid, mutation), result in zip(task_info, results):
        key = f"{pmid}:{mutation}"
        
        if isinstance(result, Exception):
            stats.failed += 1
            append_jsonl(FAILED_FILE, {
                "pmid": pmid,
                "mutation": mutation,
                "error": str(result),
                "timestamp": datetime.now().isoformat()
            })
        elif result:
            append_jsonl(OUTPUT_FILE, result)
            stats.successful += 1
            stats.total_qa_pairs += len(result["qa_pairs"])
        else:
            stats.failed += 1
            append_jsonl(ERROR_FILE, {
                "pmid": pmid,
                "mutation": mutation,
                "error": "no_valid_qa_pairs"
            })
        
        processed_keys.add(key)
        stats.processed += 1


async def main_async(args):
    """Main async processing loop."""
    global shutdown_requested
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load data
    logger.info("Loading data...")
    with open(INPUT_VERIFIED, "r") as f:
        verified = json.load(f)
    with open(INPUT_ARTICLES, "r") as f:
        articles = json.load(f)

    mismatched_pmids = set()
    if os.path.exists(MISMATCHED_PMIDS_FILE):
        with open(MISMATCHED_PMIDS_FILE, 'r') as f:
            mismatched_pmids = set(json.load(f))
        logger.info(f"Loaded {len(mismatched_pmids):,} mismatched PMIDs to exclude")
    
    # Setup output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint or processed keys
    if args.resume:
        processed_keys, saved_stats = load_checkpoint()
        if processed_keys:
            logger.info(f"Resume mode: loaded {len(processed_keys):,} from checkpoint")
        else:
            processed_keys = load_processed_from_output()
            logger.info(f"Resume mode: no checkpoint found, loaded {len(processed_keys):,} from output file")
    else:
        processed_keys = set()
    
    # Build list of mutations to process
    to_process = []
    stats = ProcessingStats()
    
    for pmid, content in verified.items():
        if pmid in mismatched_pmids:
            stats.skipped_mismatched_pmids += len(content.get("verified_mutations", []))
            continue
        mutations = content.get("verified_mutations", [])
        if not mutations:
            continue
        
        article_text = articles.get(pmid, {}).get("article", "")
        if not article_text:
            continue
        
        sentences = split_into_sentences(article_text)
        
        for mut in mutations:
            mutation = mut["mutation"]
            key = f"{pmid}:{mutation}"
            
            # Skip already processed
            if key in processed_keys:
                stats.skipped_already_processed += 1
                continue
            
            # Skip rsIDs and gene names
            if not is_usable_mutation(mutation):
                stats.skipped_rsid += 1
                continue
            
            consequence = mut.get("consequence_summary") or ""
            if len(consequence) < 50:
                stats.skipped_short_consequence += 1
                continue
            
            evidence_ids = mut.get("evidence_sentences", [])
            evidence = filter_evidence_sentences(sentences, evidence_ids)
            
            if not evidence:
                stats.skipped_no_evidence += 1
                continue
            
            to_process.append((pmid, mutation, consequence, evidence))
            stats.total_mutations += 1
    
    logger.info(f"=== PRE-FILTERING SUMMARY ===")
    logger.info(f"Total mutations to process: {stats.total_mutations:,}")
    logger.info(f"Skipped (already processed): {stats.skipped_already_processed:,}")
    logger.info(f"Skipped (mismatched PMID): {stats.skipped_mismatched_pmids:,}") 
    logger.info(f"Skipped (rsID/gene name): {stats.skipped_rsid:,}")
    logger.info(f"Skipped (short consequence): {stats.skipped_short_consequence:,}")
    logger.info(f"Skipped (no valid evidence): {stats.skipped_no_evidence:,}")
    
    if args.limit:
        to_process = to_process[:args.limit]
        logger.info(f"Limited to {args.limit} mutations")
    
    if not to_process:
        logger.info("Nothing to process!")
        return
    
    # Initialize client
    client = OpenRouterClient(args.api_key, args.model)
    client.semaphore = asyncio.Semaphore(args.workers)
    
    # Process in batches
    batch_size = args.workers * 2
    start_time = time.time()
    last_checkpoint = 0
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(to_process), batch_size):
            if shutdown_requested:
                logger.info("Shutdown requested, saving progress...")
                break
            
            batch = to_process[i:i + batch_size]
            await process_batch(client, session, batch, stats, processed_keys)
            
            # Checkpoint
            if stats.processed - last_checkpoint >= CHECKPOINT_INTERVAL:
                save_checkpoint(processed_keys, asdict(stats))
                last_checkpoint = stats.processed
            
            # Progress logging
            if stats.processed % 500 == 0:
                elapsed = time.time() - start_time
                rate = stats.processed / elapsed if elapsed > 0 else 0
                remaining = stats.total_mutations - stats.processed
                eta = remaining / rate if rate > 0 else 0
                
                logger.info(
                    f"Progress: {stats.processed:,}/{stats.total_mutations:,} "
                    f"({stats.processed/stats.total_mutations*100:.1f}%) | "
                    f"QA pairs: {stats.total_qa_pairs:,} | "
                    f"Rate: {rate:.1f}/s | "
                    f"ETA: {eta/60:.1f}min"
                )
    
    # Final checkpoint
    save_checkpoint(processed_keys, asdict(stats))
    
    # Final stats
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PHASE 1 COMPLETE" if not shutdown_requested else "PHASE 1 INTERRUPTED")
    logger.info("=" * 60)
    logger.info(f"Total processed: {stats.processed:,}")
    logger.info(f"Successful: {stats.successful:,}")
    logger.info(f"Failed: {stats.failed:,}")
    logger.info(f"Total QA pairs generated: {stats.total_qa_pairs:,}")
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info(f"Output: {OUTPUT_FILE}")
    
    if shutdown_requested:
        logger.info("Run with --resume to continue from checkpoint")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Generate QA pairs from verified mutations")
    parser.add_argument("--model", default=DEFAULT_PHASE1_MODEL, help="OpenRouter model")
    parser.add_argument("--workers", type=int, default=100, help="Max concurrent requests")
    parser.add_argument("--limit", type=int, help="Limit mutations to process (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint/output file")
    parser.add_argument("--input-verified", default=None, help="Override verified mutations input path")
    parser.add_argument("--input-articles", default=None, help="Override articles input path")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key"
    )

    args = parser.parse_args()

    # Apply path overrides
    if args.input_verified:
        global INPUT_VERIFIED
        INPUT_VERIFIED = args.input_verified
    if args.input_articles:
        global INPUT_ARTICLES
        INPUT_ARTICLES = args.input_articles
    if args.output_dir:
        global OUTPUT_DIR, OUTPUT_FILE, ERROR_FILE, CHECKPOINT_FILE, FAILED_FILE
        OUTPUT_DIR = args.output_dir
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase1_qa_pairs.jsonl")
        ERROR_FILE = os.path.join(OUTPUT_DIR, "phase1_errors.jsonl")
        CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")
        FAILED_FILE = os.path.join(OUTPUT_DIR, "failed_mutations.jsonl")

    if not args.api_key:
        logger.error("No API key. Set OPENROUTER_API_KEY or use --api-key")
        sys.exit(1)

    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()