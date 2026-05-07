#!/usr/bin/env python3
"""
Phase 3: Cross-validation judge for proteomics pipeline.

For each QA pair from Phase 2:
- Test 1 (R2 → A1): Does Phase 2's rationale support Phase 1's answer?
- Test 2 (R1 → A2): Does Phase 1's rationale support Phase 2's answer?

Classification:
- cross-grounded: Both tests pass
- grounded: Only R2→A1 passes
- ungrounded: Neither passes

Usage:
    python phase3_judge.py --workers 50
    python phase3_judge.py --resume
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
from typing import Dict, List, Optional, Tuple

# ============================================================================
# CONFIG
# ============================================================================

INPUT_PHASE2 = "./claim_extractor/proteomics/phase_2/output/phase2_qa_pairs.jsonl"
OUTPUT_DIR = "./claim_extractor/proteomics/phase_3/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase3_validated.jsonl")
ERROR_FILE = os.path.join(OUTPUT_DIR, "phase3_errors.jsonl")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")
STATS_FILE = os.path.join(OUTPUT_DIR, "stats.json")

DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"
CHECKPOINT_INTERVAL = 100

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

JUDGE_SYSTEM_PROMPT = """You are evaluating whether a rationale supports an answer to a question about a protein.

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

JUDGE_USER_PROMPT = """PROTEIN: {protein}

QUESTION: {question}

ANSWER: {answer}

RATIONALE: {rationale}

Does this rationale support this answer? Output JSON only."""

# ============================================================================
# CLIENT
# ============================================================================

class JudgeClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.semaphore = None

    async def judge(
        self,
        session: aiohttp.ClientSession,
        protein: str,
        question: str,
        answer: str,
        rationale: str,
    ) -> Tuple[Optional[bool], str, Optional[str]]:
        """Judge if rationale supports answer. Returns (supported, reasoning, error)."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/proteomics-qa-generator",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_PROMPT.format(
                    protein=protein, question=question, answer=answer, rationale=rationale
                )},
            ],
            "max_tokens": 300,
            "temperature": 0.1,
        }

        for attempt in range(3):
            try:
                async with self.semaphore:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            content = data["choices"][0]["message"]["content"].strip()
                            if content.startswith("```"):
                                content = "\n".join(l for l in content.split("\n") if not l.startswith("```"))
                            try:
                                parsed = json.loads(content)
                                return parsed.get("supported"), parsed.get("reasoning", ""), None
                            except:
                                return None, "", f"Parse error: {content[:100]}"
                        elif resp.status == 429:
                            await asyncio.sleep(5 * (attempt + 1))
                        else:
                            text = await resp.text()
                            return None, "", f"HTTP {resp.status}: {text[:100]}"
            except asyncio.TimeoutError:
                await asyncio.sleep(2)
            except Exception as e:
                return None, "", str(e)

        return None, "", "Max retries exceeded"


async def process_item(client: JudgeClient, session: aiohttp.ClientSession, item: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    """Process single item through both judge tests."""

    protein = item["protein"]
    question = item["question"]

    # Test 1: R2 -> A1 (Does Phase 2's rationale support Phase 1's answer?)
    r2_a1_supported, r2_a1_reasoning, err1 = await client.judge(
        session, protein, question,
        item["phase1_answer"],
        item["phase2_rationale"]
    )
    if err1:
        return None, f"R2->A1: {err1}"

    # Test 2: R1 -> A2 (Does Phase 1's rationale support Phase 2's answer?)
    r1_a2_supported, r1_a2_reasoning, err2 = await client.judge(
        session, protein, question,
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
    return f"{item['pmid']}|{item['protein']}|{item['question'][:80]}"


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
                        # Minimal check - just ensure required fields exist
                        if (item.get("phase1_answer") and item.get("phase2_answer") and
                            item.get("phase1_rationale") and item.get("phase2_rationale")):
                            items.append(item)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return items


async def main():
    parser = argparse.ArgumentParser(description="Phase 3: Cross-validation judge for proteins")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit items to process")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        logger.error("No API key. Set OPENROUTER_API_KEY or use --api-key")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load checkpoint
    processed_keys = load_checkpoint() if args.resume else set()
    logger.info(f"Starting with {len(processed_keys)} already processed")

    # Also check output file for already processed
    if args.resume and os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    key = make_key(obj)
                    processed_keys.add(key)
                except:
                    continue
        logger.info(f"Found {len(processed_keys)} in output file")

    stats = {
        "cross-grounded": 0,
        "grounded": 0,
        "ungrounded": 0,
        "errors": 0,
    }

    # Load existing stats if resuming
    if args.resume and os.path.exists(STATS_FILE):
        with open(STATS_FILE) as f:
            stats = json.load(f)

    # Load items to process
    items = load_phase2_items(processed_keys)
    logger.info(f"Loaded {len(items)} items to process")

    if args.limit:
        items = items[:args.limit]
        logger.info(f"Limited to {args.limit} items")

    if not items:
        logger.info("Nothing to process!")
        return

    client = JudgeClient(args.api_key, args.model)
    client.semaphore = asyncio.Semaphore(args.workers)

    start_time = time.time()
    total_processed = 0

    async with aiohttp.ClientSession() as session:
        batch_size = args.workers

        for batch_start in range(0, len(items), batch_size):
            if shutdown_requested:
                break

            batch = items[batch_start:batch_start + batch_size]

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

            total_processed += len(batch)

            # Progress
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            total_grounded = stats["cross-grounded"] + stats["grounded"]
            total_all = total_grounded + stats["ungrounded"]
            grounded_pct = 100 * total_grounded / total_all if total_all > 0 else 0

            logger.info(
                f"Progress: {total_processed}/{len(items)} ({100*total_processed/len(items):.1f}%) | "
                f"Cross: {stats['cross-grounded']} | Grounded: {stats['grounded']} | "
                f"Ungrounded: {stats['ungrounded']} | Rate: {rate:.1f}/s"
            )

            # Checkpoint
            if total_processed % CHECKPOINT_INTERVAL < batch_size:
                save_checkpoint(processed_keys, stats)

    # Final save
    save_checkpoint(processed_keys, stats)

    elapsed = time.time() - start_time
    total_grounded = stats["cross-grounded"] + stats["grounded"]
    total_all = total_grounded + stats["ungrounded"]

    logger.info("=" * 60)
    logger.info("PHASE 3 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total processed: {total_processed}")
    logger.info(f"Cross-grounded: {stats['cross-grounded']}")
    logger.info(f"Grounded: {stats['grounded']}")
    logger.info(f"Ungrounded: {stats['ungrounded']}")
    logger.info(f"Errors: {stats['errors']}")
    if total_all > 0:
        logger.info(f"Grounding rate: {100*total_grounded/total_all:.1f}%")
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
