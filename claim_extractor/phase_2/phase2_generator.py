#!/usr/bin/env python3
"""
Phase 2: Generate independent answers from LLM2 with expanded context.

For each QA pair from Phase 1:
- Extract the question
- Build a ±5 sentence window around mutation mentions in the article
- Ask LLM2 to answer the question using ONLY this expanded context
- LLM2 does NOT see LLM1's answer, consequence_summary, or original evidence

This allows LLM2 to potentially diverge from LLM1, enabling Phase 3 cross-validation.

Usage:
    python phase2_generator.py --model google/gemini-2.5-flash-lite --workers 100
    python phase2_generator.py --resume
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
from typing import Dict, List, Optional, Set, Tuple

# Add parent directory for config, and phase_0 for prefilter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase_0"))
from config import (
    PHASE1_QA_FILE, ARTICLES_FILE, PHASE2_OUTPUT_DIR, DEFAULT_PHASE2_MODEL,
)
from openrouter import OpenRouterClient
from prefilter import split_into_sentences, is_garbage_sentence

# ============================================================================
# CONFIG
# ============================================================================

INPUT_PHASE1 = str(PHASE1_QA_FILE)
INPUT_ARTICLES = str(ARTICLES_FILE)
OUTPUT_DIR = str(PHASE2_OUTPUT_DIR)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase2_qa_pairs.jsonl")
ERROR_FILE = os.path.join(OUTPUT_DIR, "phase2_errors.jsonl")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")

CONTEXT_WINDOW = 5  # ±5 sentences around mutation mentions
CHECKPOINT_INTERVAL = 1000

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    logger.info("Shutdown requested, finishing current batch...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# CONTEXT BUILDING
# ============================================================================

def find_mutation_sentence_indices(sentences: List[str], mutation: str) -> Set[int]:
    """Find which sentence indices contain the mutation."""
    indices = set()
    escaped = re.escape(mutation)
    
    for i, sent in enumerate(sentences):
        if re.search(escaped, sent, re.IGNORECASE):
            indices.add(i)
    
    return indices


def build_expanded_context(
    sentences: List[str], 
    mutation: str, 
    window: int = CONTEXT_WINDOW
) -> Tuple[List[Dict], Set[int]]:
    """
    Build expanded context window around mutation mentions.
    
    Returns:
        - List of {"id": int, "text": str} for context sentences
        - Set of sentence indices where mutation appears
    """
    mutation_indices = find_mutation_sentence_indices(sentences, mutation)
    
    if not mutation_indices:
        # Fallback: return first 20 sentences if mutation not found
        logger.warning(f"Mutation '{mutation}' not found in sentences, using fallback")
        return [{"id": i, "text": s} for i, s in enumerate(sentences[:20])], set()
    
    # Build window around all mentions
    context_indices = set()
    for idx in mutation_indices:
        start = max(0, idx - window)
        end = min(len(sentences), idx + window + 1)
        context_indices.update(range(start, end))
    
    # Sort and build context
    sorted_indices = sorted(context_indices)
    context = [{"id": i, "text": sentences[i]} for i in sorted_indices]
    
    return context, mutation_indices


def format_context_for_prompt(context: List[Dict]) -> str:
    """Format context sentences for the prompt."""
    lines = []
    for item in context:
        lines.append(f"[{item['id']}] {item['text']}")
    return "\n".join(lines)


def filter_context(context: List[Dict]) -> List[Dict]:
    """Remove garbage sentences from context."""
    return [c for c in context if not is_garbage_sentence(c['text'])]


# ============================================================================
# PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are a scientific QA system. Answer the question about the specified mutation using ONLY the provided context.

RULES:
1. Use ONLY information from the context - no external knowledge
2. If the answer is in the context, provide it - even if stated indirectly or as part of a group
3. Quote the relevant text and cite sentence numbers [N]
4. Standard mutation notation like "c.1033C>T, p.(Arg345Trp)" means c.1033C>T causes the p.Arg345Trp amino acid change

OUTPUT FORMAT (JSON only):
{
  "answer": "Your answer including the mutation name",
  "rationale": "Quote from context [N]. Brief explanation.",
  "evidence_sentence_ids": [N],
  "confidence": "high|medium|low",
  "answerable": true
}

Only set "answerable": false if the context truly contains NO relevant information about the mutation."""
USER_PROMPT_TEMPLATE = """Answer this question about the mutation {mutation}:

QUESTION: {question}

CONTEXT FROM RESEARCH PAPER:
{context}

Remember:
- Use ONLY the information in the context above
- Quote specific phrases and cite sentence numbers [N]
- Include "{mutation}" in your answer
- If the context doesn't contain enough information, say so

Output JSON only."""



# ============================================================================
# RESPONSE PARSING
# ============================================================================

def parse_llm_response(response: str) -> Optional[Dict]:
    """Parse LLM JSON response."""
    if not response:
        return None
    
    response = response.strip()
    
    # Remove markdown code blocks if present
    if response.startswith("```"):
        lines = response.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        response = "\n".join(lines)
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON object in response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    
    return None


# ============================================================================
# MAIN PROCESSING
# ============================================================================

async def process_qa_pair(
    client: OpenRouterClient,
    session: aiohttp.ClientSession,
    qa_item: Dict,
    sentences: List[str],
) -> Tuple[Optional[Dict], Optional[str]]:
    """Process a single QA pair through Phase 2."""
    
    mutation = qa_item["mutation"]
    question = qa_item["question"]
    
    # Build expanded context
    context, mutation_indices = build_expanded_context(sentences, mutation)
    context = filter_context(context)
    
    if not context:
        return None, "No valid context sentences"
    
    context_text = format_context_for_prompt(context)
    
    # Build prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        mutation=mutation,
        question=question,
        context=context_text
    )
    
    # Call LLM2
    response, error = await client.complete(session, SYSTEM_PROMPT, user_prompt)
    
    if error:
        return None, error
    
    # Parse response
    parsed = parse_llm_response(response)
    if not parsed:
        return None, f"Failed to parse response: {response[:200]}"
    
    result = {
        "pmid": qa_item["pmid"],
        "mutation": mutation,
        "question": question,
        "question_type": qa_item["question_type"],
        # Phase 1 data
        "phase1_answer": qa_item["phase1_answer"],
        "phase1_rationale": qa_item["phase1_rationale"],
        "phase1_evidence_ids": qa_item["phase1_evidence_ids"],
        # Phase 2 data
        "phase2_answer": parsed.get("answer", ""),
        "phase2_rationale": parsed.get("rationale", ""),
        "phase2_evidence_ids": parsed.get("evidence_sentence_ids", []),
        "phase2_confidence": parsed.get("confidence", "unknown"),
        "phase2_answerable": parsed.get("answerable", True),
        # Context metadata
        "context_sentence_ids": [c["id"] for c in context],
        "mutation_sentence_ids": list(mutation_indices),
    }
    
    return result, None


async def main():
    parser = argparse.ArgumentParser(description="Phase 2: Independent answer generation")
    parser.add_argument("--model", default=DEFAULT_PHASE2_MODEL)
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit QA pairs to process")
    parser.add_argument("--input-phase1", default=None, help="Override Phase 1 input path")
    parser.add_argument("--input-articles", default=None, help="Override articles input path")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    # Apply path overrides
    global INPUT_PHASE1, INPUT_ARTICLES, OUTPUT_DIR, OUTPUT_FILE, ERROR_FILE, CHECKPOINT_FILE
    if args.input_phase1:
        INPUT_PHASE1 = args.input_phase1
    if args.input_articles:
        INPUT_ARTICLES = args.input_articles
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase2_qa_pairs.jsonl")
        ERROR_FILE = os.path.join(OUTPUT_DIR, "phase2_errors.jsonl")
        CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Phase 1 results first to get needed PMIDs
    logger.info("Loading Phase 1 results...")
    qa_pairs_to_process = []
    needed_pmids = set()

    with open(INPUT_PHASE1) as f:
        for line in f:
            entry = json.loads(line)
            pmid = entry["pmid"]
            mutation = entry["mutation"]
            needed_pmids.add(pmid)

            for qa in entry.get("qa_pairs", []):
                qa_pairs_to_process.append({
                    "pmid": pmid,
                    "mutation": mutation,
                    "question": qa["question"],
                    "question_type": qa.get("question_type", "unknown"),
                    "phase1_answer": qa["answer"],
                    "phase1_rationale": qa.get("rationale", ""),
                    "phase1_evidence_ids": qa.get("evidence_sentence_ids", []),
                })

    logger.info(f"Loaded {len(qa_pairs_to_process):,} QA pairs from Phase 1 ({len(needed_pmids):,} unique PMIDs)")

    # Load only needed articles
    logger.info(f"Loading articles for {len(needed_pmids):,} PMIDs...")
    with open(INPUT_ARTICLES) as f:
        all_articles = json.load(f)

    articles = {pmid: all_articles[pmid] for pmid in needed_pmids if pmid in all_articles}
    logger.info(f"Found {len(articles):,} articles")

    # Pre-split sentences for needed articles only
    logger.info("Pre-splitting sentences for needed articles...")
    article_sentences = {}
    for pmid, article_data in articles.items():
        text = article_data.get("article") or ""
        article_sentences[pmid] = split_into_sentences(text) if text else []
    logger.info(f"Prepared sentences for {len(article_sentences):,} articles")

    # Resume logic
    processed_keys = set()
    if args.resume and os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)
            processed_keys = set(checkpoint.get("processed_keys", []))
        logger.info(f"Resuming: {len(processed_keys):,} already processed")
    
    # Filter already processed
    def make_key(qa):
        return f"{qa['pmid']}|{qa['mutation']}|{qa['question'][:80]}"
    
    qa_pairs_to_process = [
        qa for qa in qa_pairs_to_process
        if make_key(qa) not in processed_keys
    ]
    
    if args.limit:
        qa_pairs_to_process = qa_pairs_to_process[:args.limit]
    
    logger.info(f"Processing {len(qa_pairs_to_process):,} QA pairs")
    
    if not qa_pairs_to_process:
        logger.info("Nothing to process!")
        return
    
    # Initialize client
    client = OpenRouterClient(api_key, args.model)
    client.semaphore = asyncio.Semaphore(args.workers)
    
# Process with progress tracking
    total_results = 0
    total_errors = 0
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        # Process in batches for true concurrency
        batch_size = args.workers
        
        for batch_start in range(0, len(qa_pairs_to_process), batch_size):
            if shutdown_requested:
                logger.info("Shutdown requested, saving checkpoint...")
                break
            
            batch = qa_pairs_to_process[batch_start:batch_start + batch_size]
            
            # Create tasks for this batch
            tasks = []
            for qa in batch:
                sentences = article_sentences.get(qa["pmid"], [])
                task = process_qa_pair(client, session, qa, sentences)
                tasks.append((qa, task))
            
            # Run batch concurrently
            results = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            # Process results
            for (qa, _), result in zip(tasks, results):
                key = make_key(qa)
                processed_keys.add(key)
                
                if isinstance(result, Exception):
                    total_errors += 1
                    with open(ERROR_FILE, "a") as f:
                        f.write(json.dumps({"qa": qa, "error": str(result)}) + "\n")
                elif result[0] is not None:
                    total_results += 1
                    with open(OUTPUT_FILE, "a") as f:
                        f.write(json.dumps(result[0]) + "\n")
                else:
                    total_errors += 1
                    with open(ERROR_FILE, "a") as f:
                        f.write(json.dumps({"qa": qa, "error": result[1]}) + "\n")
            
            # Progress logging
            processed = batch_start + len(batch)
            if processed % 500 < batch_size or batch_start + batch_size >= len(qa_pairs_to_process):
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(qa_pairs_to_process) - processed) / rate / 60 if rate > 0 else 0
                logger.info(
                    f"Progress: {processed:,}/{len(qa_pairs_to_process):,} ({100*processed/len(qa_pairs_to_process):.1f}%) | "
                    f"Results: {total_results:,} | Errors: {total_errors:,} | "
                    f"Rate: {rate:.1f}/s | ETA: {eta:.1f}min"
                )
            
            # Checkpoint
            if (batch_start + batch_size) % CHECKPOINT_INTERVAL < batch_size:
                with open(CHECKPOINT_FILE, "w") as f:
                    json.dump({"processed_keys": list(processed_keys)}, f)
    
    # Final checkpoint
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed_keys": list(processed_keys)}, f)
    
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PHASE 2 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total processed: {total_results + total_errors:,}")
    logger.info(f"Successful: {total_results:,}")
    logger.info(f"Errors: {total_errors:,}")
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info(f"Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())