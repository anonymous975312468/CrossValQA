#!/usr/bin/env python3
"""
Phase 2: Generate independent answers from LLM2 with expanded context.

Adapted for proteomics pipeline - searches for proteins instead of mutations.

For each QA pair from Phase 1:
- Extract the question
- Build a ±5 sentence window around protein mentions in the article
- Searches for protein name AND synonyms (canonical name, gene symbol)
- Ask LLM2 to answer the question using ONLY this expanded context
- LLM2 does NOT see LLM1's answer, consequence_summary, or original evidence

This allows LLM2 to potentially diverge from LLM1, enabling Phase 3 cross-validation.

Usage:
    python phase2_generator.py --model google/gemini-2.5-flash-lite --workers 50
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

# Add prefilter module for sentence splitting
sys.path.insert(0, "./claim_extractor/phase_0")
from prefilter import split_into_sentences

# ============================================================================
# CONFIG
# ============================================================================

INPUT_PHASE1 = "./claim_extractor/proteomics/phase_1/output/phase1_qa_pairs.jsonl"
INPUT_ARTICLES = "./claim_extractor/proteomics/proteomics_dataset/all_articles.json"
OUTPUT_DIR = "./claim_extractor/proteomics/phase_2/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase2_qa_pairs.jsonl")
ERROR_FILE = os.path.join(OUTPUT_DIR, "phase2_errors.jsonl")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")

CONTEXT_WINDOW = 5  # ±5 sentences around protein mentions
CHECKPOINT_INTERVAL = 100  # Smaller dataset, checkpoint more frequently

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

def find_protein_sentence_indices(sentences: List[str], protein: str, synonyms: List[str] = None) -> Set[int]:
    """
    Find which sentence indices contain the protein or its synonyms.

    Args:
        sentences: List of sentences from article
        protein: Primary protein name
        synonyms: Additional names to search for (canonical name, gene symbol, etc.)

    Returns:
        Set of sentence indices containing any of the protein names
    """
    indices = set()

    # Build list of all names to search for
    search_terms = [protein]
    if synonyms:
        search_terms.extend([s for s in synonyms if s and s != protein])

    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in search_terms:
        term_lower = term.lower()
        if term_lower not in seen:
            seen.add(term_lower)
            unique_terms.append(term)

    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        for term in unique_terms:
            # Use word boundary matching for short terms to avoid false positives
            if len(term) <= 4:
                # For short terms like "AGP", "IL-6", use word boundaries
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, sent, re.IGNORECASE):
                    indices.add(i)
                    break
            else:
                # For longer terms, simple case-insensitive search
                if term.lower() in sent_lower:
                    indices.add(i)
                    break

    return indices


def build_expanded_context(
    sentences: List[str],
    protein: str,
    synonyms: List[str] = None,
    window: int = CONTEXT_WINDOW
) -> Tuple[List[Dict], Set[int]]:
    """
    Build expanded context window around protein mentions.

    Returns:
        - List of {"id": int, "text": str} for context sentences
        - Set of sentence indices where protein appears
    """
    protein_indices = find_protein_sentence_indices(sentences, protein, synonyms)

    if not protein_indices:
        # Fallback: return first 20 sentences if protein not found
        logger.warning(f"Protein '{protein}' not found in sentences, using fallback")
        return [{"id": i, "text": s} for i, s in enumerate(sentences[:20])], set()

    # Build window around all mentions
    context_indices = set()
    for idx in protein_indices:
        start = max(0, idx - window)
        end = min(len(sentences), idx + window + 1)
        context_indices.update(range(start, end))

    # Sort and build context
    sorted_indices = sorted(context_indices)
    context = [{"id": i, "text": sentences[i]} for i in sorted_indices]

    return context, protein_indices


def format_context_for_prompt(context: List[Dict]) -> str:
    """Format context sentences for the prompt."""
    lines = []
    for item in context:
        lines.append(f"[{item['id']}] {item['text']}")
    return "\n".join(lines)


# ============================================================================
# GARBAGE FILTER
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


def filter_context(context: List[Dict]) -> List[Dict]:
    """Remove garbage sentences from context."""
    return [c for c in context if not is_garbage_sentence(c['text'])]


# ============================================================================
# PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are a scientific QA system specializing in proteomics. Answer the question about the specified protein using ONLY the provided context.

RULES:
1. Use ONLY information from the context - no external knowledge
2. If the answer is in the context, provide it - even if stated indirectly
3. Quote the relevant text and cite sentence numbers [N]
4. Include the protein name in your answer

OUTPUT FORMAT (JSON only):
{
  "answer": "Your answer including the protein name",
  "rationale": "Quote from context [N]. Brief explanation.",
  "evidence_sentence_ids": [N],
  "confidence": "high|medium|low",
  "answerable": true
}

Only set "answerable": false if the context truly contains NO relevant information about the protein."""

USER_PROMPT_TEMPLATE = """Answer this question about the protein {protein}:

QUESTION: {question}

CONTEXT FROM RESEARCH PAPER:
{context}

Remember:
- Use ONLY the information in the context above
- Quote specific phrases and cite sentence numbers [N]
- Include "{protein}" in your answer
- If the context doesn't contain enough information, say so

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
        max_tokens: int = 1000,
        temperature: float = 0.3,
        retries: int = 3
    ) -> Tuple[Optional[str], Optional[str]]:
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
            try:
                async with self.semaphore:
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
                        else:
                            text = await resp.text()
                            last_error = f"HTTP {resp.status}: {text[:200]}"
            except asyncio.TimeoutError:
                last_error = "Timeout"
            except Exception as e:
                last_error = str(e)

            await asyncio.sleep(1 * (attempt + 1))

        return None, last_error


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
# ARTICLE LOADING
# ============================================================================

def load_articles(path: str) -> Dict[str, Dict]:
    """Load articles from proteomics all_articles.json format."""
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
    if article.get("full_text"):
        return article["full_text"]

    if article.get("sections"):
        sections_text = []
        for section in article["sections"]:
            if isinstance(section, dict) and section.get("text"):
                sections_text.append(section["text"])
            elif isinstance(section, str):
                sections_text.append(section)
        if sections_text:
            return " ".join(sections_text)

    return article.get("abstract", "")


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

    protein = qa_item["protein"]
    question = qa_item["question"]
    synonyms = qa_item.get("synonyms", [])

    # Build expanded context with synonym search
    context, protein_indices = build_expanded_context(sentences, protein, synonyms)
    context = filter_context(context)

    if not context:
        return None, "No valid context sentences"

    context_text = format_context_for_prompt(context)

    # Build prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        protein=protein,
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
        "protein": protein,
        "gene_id": qa_item.get("gene_id"),
        "uniprot_accession": qa_item.get("uniprot_accession"),
        "uniprot_entry_name": qa_item.get("uniprot_entry_name"),
        "protein_name_canonical": qa_item.get("protein_name_canonical"),
        "question": question,
        "question_type": qa_item.get("question_type", "unknown"),
        # Phase 1 data
        "phase1_answer": qa_item["phase1_answer"],
        "phase1_rationale": qa_item.get("phase1_rationale", ""),
        "phase1_evidence_ids": qa_item.get("phase1_evidence_ids", []),
        # Phase 2 data
        "phase2_answer": parsed.get("answer", ""),
        "phase2_rationale": parsed.get("rationale", ""),
        "phase2_evidence_ids": parsed.get("evidence_sentence_ids", []),
        "phase2_confidence": parsed.get("confidence", "unknown"),
        "phase2_answerable": parsed.get("answerable", True),
        # Context metadata
        "context_sentence_ids": [c["id"] for c in context],
        "protein_sentence_ids": list(protein_indices),
    }

    return result, None


async def main():
    parser = argparse.ArgumentParser(description="Phase 2: Independent answer generation for proteins")
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit QA pairs to process")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        logger.error("No API key. Set OPENROUTER_API_KEY or use --api-key")
        sys.exit(1)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load articles
    logger.info("Loading articles...")
    articles = load_articles(INPUT_ARTICLES)
    logger.info(f"Loaded {len(articles)} articles")

    # Pre-split sentences for all articles
    logger.info("Pre-splitting sentences...")
    article_sentences = {}
    for pmid, article in articles.items():
        text = get_article_text(article)
        article_sentences[pmid] = split_into_sentences(text) if text else []
    logger.info(f"Prepared sentences for {len(article_sentences)} articles")

    # Load Phase 1 results and flatten QA pairs
    logger.info("Loading Phase 1 results...")
    qa_pairs_to_process = []

    with open(INPUT_PHASE1) as f:
        for line in f:
            entry = json.loads(line)
            pmid = entry["pmid"]
            protein = entry["protein"]

            # Build synonyms list from available identifiers
            synonyms = []
            if entry.get("protein_name_canonical"):
                synonyms.append(entry["protein_name_canonical"])
            # Could also add gene symbol if we had it in the data

            for qa in entry.get("qa_pairs", []):
                qa_pairs_to_process.append({
                    "pmid": pmid,
                    "protein": protein,
                    "gene_id": entry.get("gene_id"),
                    "uniprot_accession": entry.get("uniprot_accession"),
                    "uniprot_entry_name": entry.get("uniprot_entry_name"),
                    "protein_name_canonical": entry.get("protein_name_canonical"),
                    "synonyms": synonyms,
                    "question": qa["question"],
                    "question_type": qa.get("question_type", "unknown"),
                    "phase1_answer": qa["answer"],
                    "phase1_rationale": qa.get("rationale", ""),
                    "phase1_evidence_ids": qa.get("evidence_sentence_ids", []),
                })

    logger.info(f"Loaded {len(qa_pairs_to_process)} QA pairs from Phase 1")

    # Resume logic
    processed_keys = set()
    if args.resume and os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)
            processed_keys = set(checkpoint.get("processed_keys", []))
        logger.info(f"Resuming: {len(processed_keys)} already processed")

    # Also check output file for already processed
    if args.resume and os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    key = f"{obj['pmid']}|{obj['protein']}|{obj['question'][:80]}"
                    processed_keys.add(key)
                except:
                    continue
        logger.info(f"Found {len(processed_keys)} in output file")

    # Filter already processed
    def make_key(qa):
        return f"{qa['pmid']}|{qa['protein']}|{qa['question'][:80]}"

    qa_pairs_to_process = [
        qa for qa in qa_pairs_to_process
        if make_key(qa) not in processed_keys
    ]

    if args.limit:
        qa_pairs_to_process = qa_pairs_to_process[:args.limit]

    logger.info(f"Processing {len(qa_pairs_to_process)} QA pairs")

    if not qa_pairs_to_process:
        logger.info("Nothing to process!")
        return

    # Initialize client
    client = OpenRouterClient(args.api_key, args.model)
    client.semaphore = asyncio.Semaphore(args.workers)

    # Process with progress tracking
    total_results = 0
    total_errors = 0
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
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
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(qa_pairs_to_process) - processed) / rate / 60 if rate > 0 else 0
            logger.info(
                f"Progress: {processed}/{len(qa_pairs_to_process)} ({100*processed/len(qa_pairs_to_process):.1f}%) | "
                f"Results: {total_results} | Errors: {total_errors} | "
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
    logger.info(f"Total processed: {total_results + total_errors}")
    logger.info(f"Successful: {total_results}")
    logger.info(f"Errors: {total_errors}")
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info(f"Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
