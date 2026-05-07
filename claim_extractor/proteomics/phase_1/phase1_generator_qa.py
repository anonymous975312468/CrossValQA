#!/usr/bin/env python3
"""
Phase 1: Generate QA pairs + rationales from verified proteins.

Adapted from mutation pipeline for proteomics/protein-focused QA generation.

Features:
- Checkpoints every N proteins
- Graceful shutdown on Ctrl+C
- Resumes from last checkpoint
- Logs failed proteins for retry

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

# Add prefilter module
sys.path.insert(0, "./claim_extractor/phase_0")
from prefilter import split_into_sentences

# ============================================================================
# CONFIG
# ============================================================================

INPUT_VERIFIED = "./claim_extractor/proteomics/phase_0/output/verified_proteins.json"
INPUT_ARTICLES = "./claim_extractor/proteomics/proteomics_dataset/all_articles.json"
OUTPUT_DIR = "./claim_extractor/proteomics/phase_1/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase1_qa_pairs.jsonl")
ERROR_FILE = os.path.join(OUTPUT_DIR, "phase1_errors.jsonl")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")
FAILED_FILE = os.path.join(OUTPUT_DIR, "failed_proteins.jsonl")


CHECKPOINT_INTERVAL = 100  # Save checkpoint every N proteins (smaller dataset)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# GARBAGE SENTENCE FILTER
# ============================================================================

GARBAGE_PATTERNS = [
    r'^All rights reserved',
    r'^Accepted Article',
    r'^This article is protected',
    r'^Author Man',
    r'^Copyright',
    r'^\s*\d+\s*$',
    r'^\s*[A-Z]{1,5}\s*$',
    r'^Table\s+\d',
    r'^Figure\s+\d',
    r'^Supplementary',
    r'^References?\s*$',
    r'^Acknowledgment',
    r'^\d+\s+g\.\d+',
]
GARBAGE_RE = re.compile('|'.join(GARBAGE_PATTERNS), re.IGNORECASE)


def is_garbage_sentence(sent: str) -> bool:
    """Check if a sentence is garbage."""
    if len(sent) < 20:
        return True
    if GARBAGE_RE.search(sent):
        return True
    alnum = sum(c.isalpha() for c in sent)
    if len(sent) > 0 and alnum < len(sent) * 0.5:
        return True
    return False


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

SYSTEM_PROMPT = """You are a scientific QA generator specializing in proteomics and biochemistry.

Your task is to generate high-quality question-answer pairs about proteins based on evidence from research papers.

CRITICAL RULES:
1. ALWAYS include the protein name in BOTH the question AND answer
2. Generate 2-3 diverse questions covering different aspects
3. Each answer must be directly supported by the provided evidence sentences
4. Each rationale must:
   - Quote at least one relevant phrase from the evidence (in quotation marks)
   - Cite the specific sentence number(s)
   - Explain HOW the quote supports the answer
5. Include at least one NEGATIVE framing if supported by evidence (e.g., "does NOT bind...", "is NOT found in...")
6. Do NOT infer information not explicitly stated in the evidence
7. Do NOT use external knowledge about the protein

QUESTION TYPES TO CONSIDER:
- Function: "What is the function of [protein]?"
- Interactions: "What does [protein] bind to or interact with?"
- Localization: "Where is [protein] localized in the cell?"
- Clinical: "What is the clinical significance of [protein]?"
- Structure: "What structural features does [protein] have?"
- PTM: "What modifications affect [protein]?"
- Expression: "How is [protein] expression regulated?"

OUTPUT FORMAT (JSON only):
{
  "qa_pairs": [
    {
      "question": "...",
      "answer": "...",
      "rationale": "...",
      "evidence_sentence_ids": [list of ints],
      "question_type": "function|interaction|localization|clinical|structure|ptm|expression|other"
    }
  ]
}"""


USER_PROMPT_TEMPLATE = """Generate 2-3 diverse QA pairs for this protein:

PROTEIN: {protein}

CORE FINDING (use as basis for answers):
{consequence_summary}

EVIDENCE SENTENCES FROM THE PAPER:
{evidence_text}

Remember:
- Include "{protein}" in every question AND answer
- Quote specific phrases from evidence in rationales
- Cite sentence numbers
- Explain how evidence supports the answer
- Try to cover different question types
- Include negative framing if appropriate

Output JSON only."""


# ============================================================================
# OPENROUTER CLIENT
# ============================================================================

class OpenRouterClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.semaphore = None

    async def complete(
        self,
        session: aiohttp.ClientSession,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1500,
        temperature: float = 0.3,
        retries: int = 3
    ) -> tuple[Optional[str], Optional[str]]:
        """Make completion request with retries."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/proteomics-qa-generator",
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
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data["choices"][0]["message"]["content"], None
                        elif resp.status == 429:
                            wait_time = 5 * (attempt + 1)
                            logger.warning(f"Rate limited, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            last_error = "rate_limited"
                        else:
                            error_text = await resp.text()
                            last_error = f"HTTP {resp.status}: {error_text[:200]}"
                            if resp.status >= 500:
                                await asyncio.sleep(2 * (attempt + 1))
                            else:
                                return None, last_error
                except asyncio.TimeoutError:
                    last_error = "timeout"
                    await asyncio.sleep(2)
                except Exception as e:
                    last_error = str(e)
                    await asyncio.sleep(2)

        return None, last_error


# ============================================================================
# QA GENERATION
# ============================================================================

def parse_llm_response(response: str, protein: str) -> Optional[List[Dict]]:
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
        protein_lower = protein.lower()
        for qa in qa_pairs:
            question = qa.get("question", "")
            answer = qa.get("answer", "")

            # Validate protein name appears in both Q and A
            if protein_lower in question.lower() and protein_lower in answer.lower():
                valid_pairs.append(qa)

        return valid_pairs if valid_pairs else None

    except (json.JSONDecodeError, KeyError) as e:
        logger.debug(f"Failed to parse response: {e}")
        return None


async def generate_qa_for_protein(
    client: OpenRouterClient,
    session: aiohttp.ClientSession,
    pmid: str,
    protein_data: Dict,
    evidence_sentences: List[Dict],
) -> Optional[Dict]:
    """Generate QA pairs for a single protein."""

    protein = protein_data["protein"]
    consequence_summary = protein_data.get("consequence_summary", "")

    evidence_text = "\n".join([
        f"[Sentence {e['id']}]: {e['text']}"
        for e in evidence_sentences
    ])

    if not evidence_text.strip():
        return None

    user_prompt = USER_PROMPT_TEMPLATE.format(
        protein=protein,
        consequence_summary=consequence_summary,
        evidence_text=evidence_text
    )

    response, error = await client.complete(session, SYSTEM_PROMPT, user_prompt)

    if error:
        return None

    qa_pairs = parse_llm_response(response, protein)

    if not qa_pairs:
        return None

    return {
        "pmid": pmid,
        "protein": protein,
        "gene_id": protein_data.get("gene_id"),
        "uniprot_accession": protein_data.get("uniprot_accession"),
        "uniprot_entry_name": protein_data.get("uniprot_entry_name"),
        "protein_name_canonical": protein_data.get("protein_name_canonical"),
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

    logger.info(f"Checkpoint saved: {len(processed_keys):,} proteins processed")


def load_checkpoint() -> tuple[Set[str], dict]:
    """Load checkpoint from disk."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set(), {}

    try:
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint = json.load(f)

        processed_keys = set(checkpoint.get("processed_keys", []))
        stats = checkpoint.get("stats", {})

        logger.info(f"Loaded checkpoint: {len(processed_keys):,} proteins already processed")
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
                key = f"{obj['pmid']}:{obj['protein']}"
                processed.add(key)
            except:
                continue

    return processed


# ============================================================================
# BATCH PROCESSING
# ============================================================================

@dataclass
class ProcessingStats:
    total_proteins: int = 0
    processed: int = 0
    successful: int = 0
    failed: int = 0
    total_qa_pairs: int = 0
    skipped_short_consequence: int = 0
    skipped_no_evidence: int = 0
    skipped_already_processed: int = 0


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
    """Process a batch of proteins concurrently."""

    tasks = []
    task_info = []
    for pmid, protein_data, evidence in batch:
        task = asyncio.create_task(
            generate_qa_for_protein(
                client, session, pmid, protein_data, evidence
            )
        )
        tasks.append(task)
        task_info.append((pmid, protein_data["protein"]))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for (pmid, protein), result in zip(task_info, results):
        key = f"{pmid}:{protein}"

        if isinstance(result, Exception):
            stats.failed += 1
            append_jsonl(FAILED_FILE, {
                "pmid": pmid,
                "protein": protein,
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
                "protein": protein,
                "error": "no_valid_qa_pairs"
            })

        processed_keys.add(key)
        stats.processed += 1


def load_articles(path: str) -> Dict[str, Dict]:
    """Load articles from all_articles.json format."""
    with open(path, "r") as f:
        data = json.load(f)

    # Convert list format to dict keyed by PMID
    articles_dict = {}
    for article in data.get("articles", []):
        pmid = article.get("pmid")
        if pmid:
            articles_dict[pmid] = article

    return articles_dict


def get_article_text(article: Dict) -> str:
    """Extract full text from article, falling back to abstract."""
    # Try full_text first
    if article.get("full_text"):
        return article["full_text"]

    # Try sections
    if article.get("sections"):
        sections_text = []
        for section in article["sections"]:
            if isinstance(section, dict):
                if section.get("text"):
                    sections_text.append(section["text"])
            elif isinstance(section, str):
                sections_text.append(section)
        if sections_text:
            return " ".join(sections_text)

    # Fall back to abstract
    return article.get("abstract", "")


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

    articles = load_articles(INPUT_ARTICLES)
    logger.info(f"Loaded {len(articles)} articles")

    # Setup output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load checkpoint or processed keys
    if args.resume:
        processed_keys = load_processed_from_output()
        logger.info(f"Resume mode: {len(processed_keys):,} already in output file")
    else:
        processed_keys = set()

    # Build list of proteins to process
    to_process = []
    stats = ProcessingStats()

    for pmid, content in verified.items():
        proteins = content.get("verified_proteins", [])
        if not proteins:
            continue

        article = articles.get(pmid, {})
        article_text = get_article_text(article)
        if not article_text:
            logger.warning(f"No article text for PMID {pmid}")
            continue

        sentences = split_into_sentences(article_text)

        for prot in proteins:
            protein = prot["protein"]
            key = f"{pmid}:{protein}"

            # Skip already processed
            if key in processed_keys:
                stats.skipped_already_processed += 1
                continue

            consequence = prot.get("consequence_summary") or ""
            if len(consequence) < 50:
                stats.skipped_short_consequence += 1
                continue

            evidence_ids = prot.get("evidence_sentences", [])
            evidence = filter_evidence_sentences(sentences, evidence_ids)

            if not evidence:
                stats.skipped_no_evidence += 1
                continue

            to_process.append((pmid, prot, evidence))
            stats.total_proteins += 1

    logger.info(f"=== PRE-FILTERING SUMMARY ===")
    logger.info(f"Total proteins to process: {stats.total_proteins:,}")
    logger.info(f"Skipped (already processed): {stats.skipped_already_processed:,}")
    logger.info(f"Skipped (short consequence): {stats.skipped_short_consequence:,}")
    logger.info(f"Skipped (no valid evidence): {stats.skipped_no_evidence:,}")

    if args.limit:
        to_process = to_process[:args.limit]
        logger.info(f"Limited to {args.limit} proteins")

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
            if stats.processed % 10 == 0 or stats.processed == stats.total_proteins:
                elapsed = time.time() - start_time
                rate = stats.processed / elapsed if elapsed > 0 else 0
                remaining = stats.total_proteins - stats.processed
                eta = remaining / rate if rate > 0 else 0

                logger.info(
                    f"Progress: {stats.processed:,}/{stats.total_proteins:,} "
                    f"({stats.processed/stats.total_proteins*100:.1f}%) | "
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
    parser = argparse.ArgumentParser(description="Phase 1: Generate QA pairs from verified proteins")
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite", help="OpenRouter model")
    parser.add_argument("--workers", type=int, default=50, help="Max concurrent requests")
    parser.add_argument("--limit", type=int, help="Limit proteins to process (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint/output file")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key"
    )

    args = parser.parse_args()

    if not args.api_key:
        logger.error("No API key. Set OPENROUTER_API_KEY or use --api-key")
        sys.exit(1)

    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
