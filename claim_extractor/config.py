"""
Central configuration for the CrossValQA claim extraction pipeline.

All paths are relative to PIPELINE_ROOT, which is auto-detected from this file's location.
Override any path by setting environment variables or editing this file.
"""

from pathlib import Path
import os

# ============================================================================
# ROOT PATHS
# ============================================================================

# Auto-detect: this file is at claim_extractor/config.py
CLAIM_EXTRACTOR_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = CLAIM_EXTRACTOR_DIR.parent

# ============================================================================
# DATABASE PATHS
# ============================================================================

DATABASE_DIR = PIPELINE_ROOT / "database"
ARTICLES_FILE = DATABASE_DIR / "articles_with_mutations.json"
VERIFIED_MUTATIONS_FILE = DATABASE_DIR / "verified_mutations.json"
EXTRACTION_STATS_FILE = DATABASE_DIR / "extraction_stats.json"
VERIFICATION_STATS_FILE = DATABASE_DIR / "verification_stats.json"
FAILED_BATCHES_FILE = DATABASE_DIR / "failed_batches.jsonl"
MISMATCHED_PMIDS_FILE = DATABASE_DIR / "mismatched_pmids.json"

# ============================================================================
# PHASE OUTPUT PATHS
# ============================================================================

PHASE0_OUTPUT_DIR = CLAIM_EXTRACTOR_DIR / "phase_0" / "output"
PHASE1_OUTPUT_DIR = CLAIM_EXTRACTOR_DIR / "phase_1" / "output"
PHASE2_OUTPUT_DIR = CLAIM_EXTRACTOR_DIR / "phase_2" / "output"
PHASE3_OUTPUT_DIR = CLAIM_EXTRACTOR_DIR / "phase_3" / "output"

PHASE1_QA_FILE = PHASE1_OUTPUT_DIR / "phase1_qa_pairs.jsonl"
PHASE2_QA_FILE = PHASE2_OUTPUT_DIR / "phase2_qa_pairs.jsonl"
PHASE3_VALIDATED_FILE = PHASE3_OUTPUT_DIR / "phase3_validated.jsonl"

# ============================================================================
# REFERENCE DATA
# ============================================================================

MAPPINGS_DIR = PIPELINE_ROOT / "mappings"
GENE_INFO_FILE = MAPPINGS_DIR / "Homo_sapiens.gene_info"

# ============================================================================
# FINAL DATASET
# ============================================================================

UNIPROT_OUTPUT_DIR = PIPELINE_ROOT / "uniprot" / "output"
FINAL_DATASET = PIPELINE_ROOT / "NeurIPS" / "dataset" / "all.jsonl"  

# ============================================================================
# LLM DEFAULTS
# ============================================================================

DEFAULT_PHASE1_MODEL = "google/gemini-2.5-flash"
DEFAULT_PHASE2_MODEL = "deepseek/deepseek-chat"
DEFAULT_PHASE3_MODEL = "meta-llama/llama-3.1-8b-instruct"

# ============================================================================
# API
# ============================================================================

def get_openrouter_api_key() -> str:
    """Get OpenRouter API key from environment."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        from dotenv import load_dotenv
        load_dotenv(PIPELINE_ROOT / ".env")
        key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError("OPENROUTER_API_KEY not set. Add to .env or export it.")
    return key


def get_ncbi_api_key() -> str:
    """Get NCBI API key from environment."""
    key = os.environ.get("NCBI_API_KEY", "")
    if not key:
        from dotenv import load_dotenv
        load_dotenv(PIPELINE_ROOT / ".env")
        key = os.environ.get("NCBI_API_KEY", "")
    return key  # optional, not required
