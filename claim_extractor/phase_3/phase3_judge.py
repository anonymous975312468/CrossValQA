"""
Phase 3: Cross-validation judge.
Safe to run in parallel with Phase 2 - processes available items and polls for new ones.
"""

import json
import os
import sys
import argparse
import asyncio
import aiohttp
import time
import logging
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory for config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PHASE2_QA_FILE, PHASE3_OUTPUT_DIR, DEFAULT_PHASE3_MODEL
from openrouter import OpenRouterClient

# ============================================================================
# CONFIG
# ============================================================================

INPUT_PHASE2 = str(PHASE2_QA_FILE)
OUTPUT_DIR = str(PHASE3_OUTPUT_DIR)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase3_validated.jsonl")
ERROR_FILE = os.path.join(OUTPUT_DIR, "phase3_errors.jsonl")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")
STATS_FILE = os.path.join(OUTPUT_DIR, "stats.json")

MODEL = DEFAULT_PHASE3_MODEL
CHECKPOINT_INTERVAL = 500
POLL_INTERVAL = 120  # seconds to wait before checking for new Phase 2 output

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    logger.info("Shutdown requested, finishing current batch...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# PROMPTS
# ============================================================================

JUDGE_SYSTEM_PROMPT = """You are evaluating whether a rationale supports an answer to a question about a genetic mutation.

Your task: Determine if the provided RATIONALE logically supports the provided ANSWER to the QUESTION.

IMPORTANT:
- Focus on whether the rationale's evidence supports the answer's conclusion
- Check that Yes/No answers match the rationale's implications
- A rationale that describes the opposite of what the answer claims means NOT SUPPORTED
- The rationale must provide evidence for the specific claim in the answer

Output JSON only:
{
  "supported": true,
  "reasoning": "Brief explanation"
}

or

{
  "supported": false,
  "reasoning": "Brief explanation"
}"""

JUDGE_USER_PROMPT = """MUTATION: {mutation}

QUESTION: {question}

ANSWER: {answer}

RATIONALE: {rationale}

Does this rationale support this answer? Output JSON only."""

# ============================================================================
# PRE-FILTER
# ============================================================================

def passes_prefilter(item: Dict) -> Tuple[bool, str]:
    """Quick checks before sending to judge."""
    mutation = item.get("mutation", "").lower()
    
    if not item.get("phase2_answerable", True):
        return False, "phase2_not_answerable"
    
    if mutation and mutation not in item.get("phase1_answer", "").lower():
        return False, "mutation_not_in_phase1_answer"
    
    if mutation and mutation not in item.get("phase2_answer", "").lower():
        return False, "mutation_not_in_phase2_answer"
    
    if not item.get("phase1_rationale"):
        return False, "missing_phase1_rationale"
    
    if not item.get("phase2_rationale"):
        return False, "missing_phase2_rationale"
    
    return True, "passed"

# ============================================================================
# JUDGE FUNCTION
# ============================================================================

async def judge_pair(
    client: OpenRouterClient,
    session: aiohttp.ClientSession,
    mutation: str,
    question: str,
    answer: str,
    rationale: str,
) -> Tuple[Optional[bool], str, Optional[str]]:
    """Judge if rationale supports answer. Returns (supported, reasoning, error)."""
    user_prompt = JUDGE_USER_PROMPT.format(
        mutation=mutation, question=question, answer=answer, rationale=rationale
    )

    content, error = await client.complete(
        session, JUDGE_SYSTEM_PROMPT, user_prompt,
        max_tokens=300, temperature=0.1, timeout=60,
    )

    if error:
        return None, "", error

    content = content.strip()
    if content.startswith("```"):
        content = "\n".join(l for l in content.split("\n") if not l.startswith("```"))

    try:
        parsed = json.loads(content)
        return parsed.get("supported"), parsed.get("reasoning", ""), None
    except Exception:
        return None, "", f"Parse error: {content[:100]}"


async def process_item(client: OpenRouterClient, session: aiohttp.ClientSession, item: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    """Process single item through both judge tests."""

    mutation = item["mutation"]
    question = item["question"]

    # Test 1: R2 -> A1
    r2_a1_supported, r2_a1_reasoning, err1 = await judge_pair(
        client, session, mutation, question,
        item["phase1_answer"],
        item["phase2_rationale"]
    )
    if err1:
        return None, f"R2->A1: {err1}"

    # Test 2: R1 -> A2
    r1_a2_supported, r1_a2_reasoning, err2 = await judge_pair(
        client, session, mutation, question,
        item["phase2_answer"],
        item["phase1_rationale"]
    )
    if err2:
        return None, f"R1->A2: {err2}"
    
    # Determine grounding
    if r2_a1_supported and r1_a2_supported:
        grounding = "cross-grounded"
    elif r2_a1_supported:
        grounding = "grounded"
    else:
        grounding = "ungrounded"
    
    result = {
        **item,
        "r2_supports_a1": r2_a1_supported,
        "r2_a1_reasoning": r2_a1_reasoning,
        "r1_supports_a2": r1_a2_supported,
        "r1_a2_reasoning": r1_a2_reasoning,
        "grounding": grounding,
    }
    
    return result, None


def load_checkpoint() -> set:
    """Load processed keys from checkpoint."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f).get("processed_keys", []))
    return set()


def save_checkpoint(processed_keys: set, stats: dict):
    """Save checkpoint and stats."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed_keys": list(processed_keys)}, f)
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)


def make_key(item: Dict) -> str:
    return f"{item['pmid']}|{item['mutation']}|{item['question'][:80]}"


def load_phase2_items(processed_keys: set) -> List[Dict]:
    """Load unprocessed items from Phase 2 output."""
    items = []
    try:
        with open(INPUT_PHASE2) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    key = make_key(item)
                    if key not in processed_keys:
                        items.append(item)
                except json.JSONDecodeError:
                    continue  # Skip incomplete lines
    except FileNotFoundError:
        pass
    return items


def passes_minimal_filter(item: Dict) -> Tuple[bool, str]:
    """Minimal checks - just ensure required fields exist for judging."""
    if not item.get("phase1_answer"):
        return False, "missing_phase1_answer"
    if not item.get("phase2_answer"):
        return False, "missing_phase2_answer"
    if not item.get("phase1_rationale"):
        return False, "missing_phase1_rationale"
    if not item.get("phase2_rationale"):
        return False, "missing_phase2_rationale"
    return True, "passed"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_PHASE3_MODEL, help="OpenRouter model for judging")
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--once", action="store_true", help="Process available items and exit (don't poll)")
    parser.add_argument("--strict-prefilter", action="store_true", help="Use strict prefilter (reject if mutation string not in answers, phase2 not answerable)")
    parser.add_argument("--input-phase2", default=None, help="Override Phase 2 input path")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    # Apply path overrides
    global INPUT_PHASE2, OUTPUT_DIR, OUTPUT_FILE, ERROR_FILE, CHECKPOINT_FILE, STATS_FILE
    if args.input_phase2:
        INPUT_PHASE2 = args.input_phase2
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase3_validated.jsonl")
        ERROR_FILE = os.path.join(OUTPUT_DIR, "phase3_errors.jsonl")
        CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")
        STATS_FILE = os.path.join(OUTPUT_DIR, "stats.json")

    # Allow CLI override of the model
    global MODEL
    MODEL = args.model
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        sys.exit(1)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load checkpoint
    processed_keys = load_checkpoint() if args.resume else set()
    logger.info(f"Starting with {len(processed_keys):,} already processed")
    if args.strict_prefilter:
        logger.info("Strict prefilter ENABLED - rejecting if mutation not in answers or phase2 not answerable")
    else:
        logger.info("Using minimal filter (only checking required fields exist)")
    
    stats = {
        "cross-grounded": 0,
        "grounded": 0,
        "ungrounded": 0,
        "errors": 0,
        "prefilter_failed": 0,
    }
    
    # Load existing stats if resuming
    if args.resume and os.path.exists(STATS_FILE):
        with open(STATS_FILE) as f:
            stats = json.load(f)
    
    client = OpenRouterClient(api_key, MODEL)
    client.semaphore = asyncio.Semaphore(args.workers)
    
    start_time = time.time()
    total_processed_this_run = 0
    
    async with aiohttp.ClientSession() as session:
        while not shutdown_requested:
            # Load available items
            items = load_phase2_items(processed_keys)
            
            # Apply prefilter (or minimal filter if --no-prefilter)
            items_to_process = []
            filter_func = passes_prefilter if args.strict_prefilter else passes_minimal_filter
            for item in items:
                passes, reason = filter_func(item)
                if passes:
                    items_to_process.append(item)
                else:
                    key = make_key(item)
                    processed_keys.add(key)
                    stats["prefilter_failed"] += 1
            
            if not items_to_process:
                if args.once:
                    logger.info("No more items to process, exiting (--once mode)")
                    break
                logger.info(f"No new items, waiting {POLL_INTERVAL}s for Phase 2...")
                save_checkpoint(processed_keys, stats)
                await asyncio.sleep(POLL_INTERVAL)
                continue
            
            logger.info(f"Processing {len(items_to_process):,} new items")
            
            # Process in batches
            batch_size = args.workers
            for batch_start in range(0, len(items_to_process), batch_size):
                if shutdown_requested:
                    break
                
                batch = items_to_process[batch_start:batch_start + batch_size]
                
                tasks = [process_item(client, session, item) for item in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for item, result in zip(batch, results):
                    key = make_key(item)
                    processed_keys.add(key)
                    
                    if isinstance(result, Exception):
                        stats["errors"] += 1
                        with open(ERROR_FILE, "a") as f:
                            f.write(json.dumps({"key": key, "error": str(result)}) + "\n")
                    elif result[0] is not None:
                        grounding = result[0]["grounding"]
                        stats[grounding] += 1
                        with open(OUTPUT_FILE, "a") as f:
                            f.write(json.dumps(result[0]) + "\n")
                    else:
                        stats["errors"] += 1
                        with open(ERROR_FILE, "a") as f:
                            f.write(json.dumps({"key": key, "error": result[1]}) + "\n")
                
                total_processed_this_run += len(batch)
                
                # Progress
                elapsed = time.time() - start_time
                rate = total_processed_this_run / elapsed if elapsed > 0 else 0
                total_grounded = stats["cross-grounded"] + stats["grounded"]
                total_all = total_grounded + stats["ungrounded"]
                grounded_pct = 100 * total_grounded / total_all if total_all > 0 else 0
                
                if total_processed_this_run % 500 < batch_size:
                    logger.info(
                        f"Processed: {total_processed_this_run:,} | "
                        f"Grounded: {total_grounded:,} ({grounded_pct:.1f}%) | "
                        f"Ungrounded: {stats['ungrounded']:,} | "
                        f"Rate: {rate:.1f}/s"
                    )
                
                # Checkpoint
                if total_processed_this_run % CHECKPOINT_INTERVAL < batch_size:
                    save_checkpoint(processed_keys, stats)
            
            save_checkpoint(processed_keys, stats)
            
            if args.once:
                break
    
    # Final save
    save_checkpoint(processed_keys, stats)
    
    elapsed = time.time() - start_time
    total_grounded = stats["cross-grounded"] + stats["grounded"]
    
    logger.info("=" * 60)
    logger.info("PHASE 3 STATUS")
    logger.info("=" * 60)
    logger.info(f"Processed this run: {total_processed_this_run:,}")
    logger.info(f"Cross-grounded: {stats['cross-grounded']:,}")
    logger.info(f"Grounded: {stats['grounded']:,}")
    logger.info(f"Ungrounded: {stats['ungrounded']:,}")
    logger.info(f"Prefilter failed: {stats['prefilter_failed']:,}")
    logger.info(f"Errors: {stats['errors']:,}")
    logger.info(f"Time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    asyncio.run(main())