#!/usr/bin/env python3
"""
LLM Protein Verification Pipeline

Verifies that proteins are functionally discussed in papers using LLM via OpenRouter.
Adapted from verify_mutations.py for proteomics use case.

Usage:
    python verify_proteomics.py --model google/gemini-2.5-flash-lite --workers 10

For small dataset (22 articles), can use defaults:
    python verify_proteomics.py
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
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path("./claim_extractor/proteomics/phase_0")
INPUT_PATH = BASE_DIR / "output/prefiltered_proteins.json"
OUTPUT_PATH = BASE_DIR / "output/verified_proteins.json"
STATS_PATH = BASE_DIR / "output/verification_stats.json"
FAILED_PATH = BASE_DIR / "output/failed_batches.jsonl"


@dataclass
class ProteinIdentifiers:
    """Container for protein database identifiers from PubTator/UniProt."""
    gene_id: Optional[str] = None  # NCBI Gene ID
    taxon_id: Optional[str] = None  # NCBI Taxonomy ID
    uniprot_accession: Optional[str] = None  # UniProt accession (e.g., Q29014)
    uniprot_entry_name: Optional[str] = None  # UniProt entry name (e.g., A1AG_PIG)
    protein_name_canonical: Optional[str] = None  # Canonical protein name from UniProt

    @classmethod
    def from_dict(cls, d: Dict) -> 'ProteinIdentifiers':
        return cls(
            gene_id=d.get("gene_id"),
            taxon_id=d.get("taxon_id"),
            uniprot_accession=d.get("uniprot_accession"),
            uniprot_entry_name=d.get("uniprot_entry_name"),
            protein_name_canonical=d.get("protein_name_canonical") or d.get("protein_name"),
        )

    def is_empty(self) -> bool:
        return not (self.gene_id or self.uniprot_accession)


@dataclass
class VerifiedProtein:
    """Result of LLM verification for a single protein."""
    protein: str
    is_discussed: bool
    consequence_summary: Optional[str]
    evidence_sentences: List[int]
    confidence: str  # high, medium, low
    identifiers: Optional[ProteinIdentifiers] = None


@dataclass
class ArticleResult:
    """Result of processing a single article."""
    pmid: str
    verified_proteins: List[VerifiedProtein]
    just_mentioned: List[str]
    all_identifiers: Optional[Dict] = None  # All identifiers found in article
    error: Optional[str] = None


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

SYSTEM_PROMPT = """You are a scientific literature analyst specializing in proteomics and biochemistry.
Your task is to determine whether proteins are substantively discussed in research papers.

A protein is "substantively discussed" if the paper describes:
- Its enzymatic activity, kinetic parameters, or catalytic mechanism
- Its structure, domains, or post-translational modifications
- Its interactions with other proteins or molecules
- Its biological function, role in pathways, or cellular localization
- Experimental characterization (purification, mass spectrometry, crystallography)
- Its relevance as a biomarker, therapeutic target, or diagnostic indicator

A protein is NOT substantively discussed if it is:
- Simply listed in a table or figure legend without explanation
- Mentioned as a housekeeping/control gene
- Referenced only as part of methodology (e.g., "anti-X antibody was used")
- Mentioned in passing in the introduction as background

CRITICAL: Your consequence_summary must ONLY include information that is DIRECTLY stated in the cited sentences.
Do not make inferences or logical deductions beyond what is explicitly written.
Every claim in your consequence_summary must be directly verifiable in the cited evidence sentences."""

USER_PROMPT_TEMPLATE = """Analyze the following proteins from a scientific paper and determine which ones are substantively discussed.

PROTEINS TO ANALYZE:
{proteins_list}

RELEVANT PAPER EXCERPTS (numbered sentences):
{numbered_sentences}

For each protein, respond in this exact JSON format:
{{
  "results": [
    {{
      "protein": "<protein_name>",
      "is_discussed": true/false,
      "consequence_summary": "<brief summary using ONLY information directly stated in cited sentences>",
      "evidence_sentences": [<list of sentence numbers that DIRECTLY support your summary>],
      "confidence": "high/medium/low"
    }}
  ]
}}

IMPORTANT RULES:
1. Only include proteins from the provided list
2. Every claim in consequence_summary must be DIRECTLY stated in at least one cited sentence
3. Do not infer function from protein name alone - only state what the paper explicitly says
4. If a protein is only mentioned in methods or as a control, mark is_discussed as false
5. Be strict - when in doubt, mark as not discussed
6. Keep consequence_summary to 1-2 sentences focusing on the key finding"""


# ============================================================================
# FAILED BATCH LOGGING
# ============================================================================

def log_failed_batch(pmid: str, proteins: List[str], error: str):
    """Append failed batch to log file for later retry."""
    FAILED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILED_PATH, "a") as f:
        f.write(json.dumps({
            "pmid": pmid,
            "proteins": proteins,
            "error": error,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }) + "\n")


# ============================================================================
# OPENROUTER API
# ============================================================================

class OpenRouterClient:
    """Async client for OpenRouter API."""

    def __init__(self, api_key: str, model: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.semaphore = None

    async def complete(
        self,
        session: aiohttp.ClientSession,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        retries: int = 3,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Make a completion request to OpenRouter with retry logic."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/proteomics-verification",
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
                            wait_time = 5 * (attempt + 1)
                            logger.warning(f"Rate limited, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            last_error = "rate_limited"
                        else:
                            error_text = await response.text()
                            last_error = f"HTTP {response.status}: {error_text[:200]}"
                            if response.status >= 500:
                                await asyncio.sleep(2 * (attempt + 1))
                            else:
                                return None, last_error
                except asyncio.TimeoutError:
                    last_error = "timeout"
                    logger.warning(f"Timeout (attempt {attempt + 1}/{retries})")
                    await asyncio.sleep(2)
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Request error: {e}")
                    await asyncio.sleep(2)

        return None, last_error


# ============================================================================
# PROTEIN VERIFICATION
# ============================================================================

def build_numbered_sentences(
    sentences: List[str],
    relevant_indices: List[int],
    context_window: int = 2
) -> str:
    """Build numbered sentences string for the prompt."""
    expanded = set()
    for idx in relevant_indices:
        for i in range(max(0, idx - context_window), min(len(sentences), idx + context_window + 1)):
            expanded.add(i)

    sorted_indices = sorted(expanded)[:100]

    lines = []
    for idx in sorted_indices:
        if idx < len(sentences):
            lines.append(f"[{idx}] {sentences[idx]}")

    return "\n".join(lines)


def parse_llm_response(response_text: str, expected_proteins: List[str], pmid: str = None) -> List[VerifiedProtein]:
    """Parse LLM JSON response into VerifiedProtein objects."""
    results = []

    try:
        # Handle markdown code blocks
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
        else:
            json_str = response_text

        data = json.loads(json_str.strip())

        for item in data.get("results", []):
            protein = item.get("protein", "")
            if protein in expected_proteins:
                results.append(VerifiedProtein(
                    protein=protein,
                    is_discussed=item.get("is_discussed", False),
                    consequence_summary=item.get("consequence_summary"),
                    evidence_sentences=item.get("evidence_sentences", []),
                    confidence=item.get("confidence", "low")
                ))
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning(f"Failed to parse LLM response for {pmid}: {e}")
        if pmid:
            log_failed_batch(pmid, expected_proteins, f"parse_error: {e}")

    return results


async def verify_protein_batch(
    client: OpenRouterClient,
    session: aiohttp.ClientSession,
    proteins: List[Dict],
    sentences: List[str],
    pmid: str = None,
) -> List[VerifiedProtein]:
    """Verify a batch of proteins using LLM."""

    proteins_list = "\n".join([f"- {p['protein']}" for p in proteins])

    # Get all relevant sentence indices
    relevant_indices = set()
    for p in proteins:
        relevant_indices.update(p.get("protein_sentences", []))
        relevant_indices.update(p.get("evidence_sentences", []))

    numbered_sentences = build_numbered_sentences(sentences, list(relevant_indices))

    user_prompt = USER_PROMPT_TEMPLATE.format(
        proteins_list=proteins_list,
        numbered_sentences=numbered_sentences
    )

    response, error = await client.complete(session, SYSTEM_PROMPT, user_prompt)

    if error:
        logger.warning(f"LLM error for {pmid}: {error}")
        protein_names = [p["protein"] for p in proteins]
        log_failed_batch(pmid, protein_names, f"llm_error: {error}")
        return []

    expected = [p["protein"] for p in proteins]
    return parse_llm_response(response, expected, pmid)


async def process_article(
    client: OpenRouterClient,
    session: aiohttp.ClientSession,
    pmid: str,
    article_data: Dict,
    batch_size: int = 5,
) -> ArticleResult:
    """Process a single article - verify prefiltered proteins."""

    try:
        likely_discussed = article_data.get("likely_discussed", [])
        just_mentioned = article_data.get("likely_just_mentioned", [])
        sentences = article_data.get("sentences", [])
        gene_annotations = article_data.get("gene_annotations", [])

        if not likely_discussed:
            return ArticleResult(
                pmid=pmid,
                verified_proteins=[],
                just_mentioned=just_mentioned,
                all_identifiers={"gene_annotations": gene_annotations},
            )

        # Build map from protein name to identifiers
        # New format: identifiers are stored directly on the protein entry
        protein_to_ids = {}
        for p in likely_discussed:
            protein_name = p.get("protein", "")
            if protein_name:
                protein_to_ids[protein_name] = ProteinIdentifiers(
                    gene_id=p.get("gene_id"),
                    taxon_id=p.get("taxon_id"),
                    uniprot_accession=p.get("uniprot_accession"),
                    uniprot_entry_name=p.get("uniprot_entry_name"),
                    protein_name_canonical=p.get("protein_name_canonical"),
                )

        # Batch proteins for LLM verification
        verified = []
        for i in range(0, len(likely_discussed), batch_size):
            batch = likely_discussed[i:i + batch_size]
            batch_results = await verify_protein_batch(client, session, batch, sentences, pmid)

            # Attach identifiers to verified proteins
            for vp in batch_results:
                if vp.protein in protein_to_ids:
                    vp.identifiers = protein_to_ids[vp.protein]

            verified.extend(batch_results)

        return ArticleResult(
            pmid=pmid,
            verified_proteins=verified,
            just_mentioned=just_mentioned,
            all_identifiers={"gene_annotations": gene_annotations},
        )

    except Exception as e:
        logger.error(f"Error processing {pmid}: {e}")
        log_failed_batch(pmid, [], f"processing_error: {e}")
        return ArticleResult(
            pmid=pmid,
            verified_proteins=[],
            just_mentioned=[],
            error=str(e)
        )


async def process_all_articles(
    data: Dict[str, Dict],
    client: OpenRouterClient,
    max_concurrent: int = 10,
    batch_size: int = 5,
) -> Dict[str, ArticleResult]:
    """Process all articles with parallel LLM calls."""

    client.semaphore = asyncio.Semaphore(max_concurrent)

    results = {}
    total = len(data)
    processed = 0
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = {}
        for pmid, article_data in data.items():
            task = asyncio.create_task(
                process_article(client, session, pmid, article_data, batch_size)
            )
            tasks[task] = pmid

        for coro in asyncio.as_completed(tasks.keys()):
            try:
                result = await coro
                results[result.pmid] = result
            except Exception as e:
                pmid = tasks.get(coro, "unknown")
                logger.error(f"Unexpected error for {pmid}: {e}")
                results[pmid] = ArticleResult(
                    pmid=pmid,
                    verified_proteins=[],
                    just_mentioned=[],
                    error=str(e)
                )

            processed += 1
            elapsed = time.time() - start_time
            logger.info(f"Processed {processed}/{total} articles ({elapsed:.1f}s)")

    return results


def serialize_verified_protein(vp: VerifiedProtein) -> Dict:
    """Serialize a VerifiedProtein to dict, handling Optional identifiers."""
    d = {
        "protein": vp.protein,
        "is_discussed": vp.is_discussed,
        "consequence_summary": vp.consequence_summary,
        "evidence_sentences": vp.evidence_sentences,
        "confidence": vp.confidence,
    }
    if vp.identifiers:
        d["gene_id"] = vp.identifiers.gene_id
        d["taxon_id"] = vp.identifiers.taxon_id
        d["uniprot_accession"] = vp.identifiers.uniprot_accession
        d["uniprot_entry_name"] = vp.identifiers.uniprot_entry_name
        d["protein_name_canonical"] = vp.identifiers.protein_name_canonical
    else:
        d["gene_id"] = None
        d["taxon_id"] = None
        d["uniprot_accession"] = None
        d["uniprot_entry_name"] = None
        d["protein_name_canonical"] = None
    return d


def save_results(results: Dict[str, ArticleResult], output_path: Path):
    """Save results in Phase 1 compatible format."""

    output = {}
    for pmid, result in results.items():
        output[pmid] = {
            "verified_proteins": [
                serialize_verified_protein(v) for v in result.verified_proteins if v.is_discussed
            ],
            "unverified_proteins": [
                serialize_verified_protein(v) for v in result.verified_proteins if not v.is_discussed
            ],
            "just_mentioned": result.just_mentioned,
            "all_identifiers": result.all_identifiers or {},
            "error": result.error,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved results to {output_path}")


def generate_stats(results: Dict[str, ArticleResult]) -> Dict:
    """Generate summary statistics."""

    total_articles = len(results)
    articles_with_verified = sum(
        1 for r in results.values()
        if any(v.is_discussed for v in r.verified_proteins)
    )
    total_verified = sum(
        len([v for v in r.verified_proteins if v.is_discussed])
        for r in results.values()
    )
    total_unverified = sum(
        len([v for v in r.verified_proteins if not v.is_discussed])
        for r in results.values()
    )
    total_just_mentioned = sum(len(r.just_mentioned) for r in results.values())
    total_errors = sum(1 for r in results.values() if r.error)

    # Count proteins with identifiers
    verified_with_ids = 0
    for r in results.values():
        for v in r.verified_proteins:
            if v.is_discussed and v.identifiers and not v.identifiers.is_empty():
                verified_with_ids += 1

    return {
        "total_articles": total_articles,
        "articles_with_verified_proteins": articles_with_verified,
        "total_verified_proteins": total_verified,
        "verified_with_identifiers": verified_with_ids,
        "total_unverified_by_llm": total_unverified,
        "total_just_mentioned": total_just_mentioned,
        "total_errors": total_errors,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Verify proteins with LLM")
    parser.add_argument(
        "--input",
        default=str(INPUT_PATH),
        help="Input JSON file (prefiltered proteins)"
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Output JSON file"
    )
    parser.add_argument(
        "--model",
        default="google/gemini-2.0-flash-001",
        help="OpenRouter model to use"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Max concurrent API requests"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Proteins per LLM call"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )

    args = parser.parse_args()

    if not args.api_key:
        logger.error("No API key provided. Set OPENROUTER_API_KEY or use --api-key")
        sys.exit(1)

    # Load data
    logger.info(f"Loading data from {args.input}")
    with open(args.input, "r") as f:
        data = json.load(f)

    articles = data.get("articles", {})
    logger.info(f"Processing {len(articles)} articles")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max concurrent requests: {args.workers}")

    # Clear previous failed batches
    if FAILED_PATH.exists():
        FAILED_PATH.unlink()

    # Initialize client
    client = OpenRouterClient(args.api_key, args.model)

    # Process
    start_time = time.time()
    try:
        results = asyncio.run(
            process_all_articles(
                articles, client,
                args.workers, args.batch_size
            )
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        results = {}
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

    total_time = time.time() - start_time

    # Save results
    save_results(results, Path(args.output))

    # Generate and save stats
    stats = generate_stats(results)
    stats["processing_time_seconds"] = total_time
    stats["model"] = args.model

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info("VERIFICATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total articles: {stats['total_articles']}")
    logger.info(f"Articles with verified proteins: {stats['articles_with_verified_proteins']}")
    logger.info(f"Total verified proteins: {stats['total_verified_proteins']}")
    logger.info(f"Total unverified by LLM: {stats['total_unverified_by_llm']}")
    logger.info(f"Total just mentioned: {stats['total_just_mentioned']}")
    logger.info(f"Errors: {stats['total_errors']}")
    logger.info(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
