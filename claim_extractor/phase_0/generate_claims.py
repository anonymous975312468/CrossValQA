#!/usr/bin/env python3
"""
**********************
****** NOT USED ******
**********************
Stage 1: Extract mutation-agnostic functional/mechanistic claims with evidence sentence IDs.

Input:  JSON file containing PubMed articles (your backfilled dataset).
Output: JSONL with one record per PubMed ID:
        - doc_id
        - did_split
        - filtered_article_char_len
        - sentences_used metadata
        - claims[] with resolved evidence (sid/start/end/text)

Notes:
- This script does NOT take a mutation list and does NOT generate QAs.
- It is designed to be paired with Stage 2 (binding) and Stage 3 (QA generation) later.
"""

import os
import re
import json
import time
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import aiohttp


# =========================
# Config
# =========================

DATA_PATH = "./external_data/pubmed_to_proteins_with_articles.backfilled.json"
OUTPUT_DIR = "./claim_extractor/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_JSONL = os.path.join(OUTPUT_DIR, "stage1_claims.jsonl")
ERROR_JSONL = os.path.join(OUTPUT_DIR, "stage1_errors.jsonl")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek-chat")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "1"))

TEMPERATURE = 0.0
MAX_TOKENS = 1600
CONTEXT_WINDOW = 10
OUTPUT_MAX_TOKENS = 1200

# Max chars of numbered sentences we send to the model (control cost)
PROMPT_MAX_CHARS = int(os.environ.get("PROMPT_MAX_CHARS", "16000"))

# Hard cap on raw text size to avoid pathological docs
MAX_RAW_CHARS = int(os.environ.get("MAX_RAW_CHARS", "2000000"))

# If we cannot split headings reliably, optionally fall back to a keyword-based reduction.
ENABLE_KEYWORD_FALLBACK = True

TEST_PMID = "31661308"

# =========================
# Section filtering (your logic)
# =========================

KEEP_SECTIONS = {
    "ABSTRACT",
    "INTRODUCTION",
    "BACKGROUND",
    "DISCUSSION",
    "CONCLUSION",
    "CONCLUSIONS",
    "OBJECTIVES",
    "METHODS",
    "MEASUREMENTS AND MAIN RESULTS",
    "RESULTS",
    "FINDINGS",
    "INTERPRETATION",
}

DROP_SECTIONS = {
    "REFERENCES",
    "BIBLIOGRAPHY",
    "ACKNOWLEDGMENTS",
    "ACKNOWLEDGEMENTS",
    "FUNDING",
    "AUTHOR CONTRIBUTIONS",
    "COMPETING INTERESTS",
    "CONFLICTS OF INTEREST",
    "SUPPLEMENTARY REFERENCES",
    "SUPPLEMENTARY MATERIAL",
    "SUPPLEMENTARY METHODS",
    "EXTENDED METHODS",
    "MATERIALS",
}

_NLP = None

def get_sentence_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP

    import spacy
    for model_name in ("en_core_sci_sm", "en_core_web_sm"):
        try:
            _NLP = spacy.load(
                model_name,
                disable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer", "ner"],
            )
            break
        except Exception:
            continue

    if _NLP is None:
        raise RuntimeError("No spaCy model available")

    if not any(p in _NLP.pipe_names for p in ("parser", "senter", "sentencizer")):
        _NLP.add_pipe("sentencizer")

    return _NLP


# A pragmatic heading splitter; if you already have SECTION_SPLIT_RE in your project,
# replace this with your exact regex.
SECTION_SPLIT_RE = re.compile(
    r"\n("
    r"(?:[A-Z][A-Z0-9 /&()\-\.,]{2,})"      # ALL CAPS style
    r"|"
    r"(?:[A-Z][a-z][A-Za-z0-9 /&()\-\.,]{1,})"  # Title Case style
    r")\n"
)

def normalize_heading(h: str) -> str:
    return re.sub(r"[^A-Z0-9 /&()\-]", "", h.upper()).strip()


def filter_article_sections(text: str) -> tuple[str, bool]:
    """
    Returns (filtered_text, did_split).
    did_split=True means we detected at least one plausible section heading and kept something.
    """
    parts = SECTION_SPLIT_RE.split(text)
    if len(parts) < 3:
        return text, False

    kept: list[str] = []
    preamble = parts[0].strip()
    if preamble:
        kept.append(preamble)

    kept_any_section = False
    for i in range(1, len(parts) - 1, 2):
        heading = parts[i].strip()

        if len(heading) > 60:
            continue

        content = parts[i + 1].strip()
        if not heading:
            continue

        norm = normalize_heading(heading)
        if norm in DROP_SECTIONS:
            continue

        if norm in KEEP_SECTIONS:
            kept_any_section = True
            if content:
                kept.append(f"\n{heading}\n{content}")

    if not kept_any_section:
        return text, False

    out = "\n".join(kept).strip()

    if len(out) < 0.3 * len(text):
        # if we kept less than 30% of the text, assume bad split
        return text, False
    
    return out, True

# Optional: very lightweight fallback if headings aren't detected.
# Keep only lines that contain mechanistic verbs/keywords (plus a little context).
MECH_KEYWORDS = [
    "increase", "decrease", "reduc", "elevat", "abolish", "impair", "enhanc",
    "no change", "unchanged", "did not affect", "failed to",
    "activity", "function", "signaling", "phosphory", "binding", "interact",
    "stability", "abundance", "degrad", "half-life", "localiz", "traffic",
    "process", "cleav", "secretion", "secreted",
]

def select_sentence_windows(sentences, keywords, window=CONTEXT_WINDOW):
    keep = [False] * len(sentences)
    for i, s in enumerate(sentences):
        low = s.text.lower()
        if any(k in low for k in keywords):
            for j in range(max(0, i - window), min(len(sentences), i + window + 1)):
                keep[j] = True
    return [sentences[i] for i in range(len(sentences)) if keep[i]]


# =========================
# Sentence segmentation + numbering
# =========================

@dataclass
class Sent:
    sid: int
    start_char: int
    end_char: int
    text: str


_SENT_SPLIT_FALLBACK = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z0-9\(\[])")

def sentence_split_spacy(text: str) -> Optional[List[Sent]]:
    try:
        nlp = get_sentence_nlp()
        doc = nlp(text)
        ...
    except Exception:
        return None

def sentence_split_fallback(text: str) -> List[Sent]:
    parts = _SENT_SPLIT_FALLBACK.split(text.strip())
    sents: List[Sent] = []
    cursor = 0
    sid = 0
    for p in parts:
        p = p.strip()
        if not p:
            continue
        idx = text.find(p, cursor)
        if idx == -1:
            idx = cursor
        st = idx
        en = idx + len(p)
        sents.append(Sent(sid=sid, start_char=st, end_char=en, text=p))
        sid += 1
        cursor = en
    return sents


def segment_sentences(text: str) -> List[Sent]:
    s = sentence_split_spacy(text)
    if s is not None and len(s) > 0:
        return s
    return sentence_split_fallback(text)


# =========================
# Stage 1 prompt
# =========================

STAGE1_SYSTEM_PROMPT = """You are extracting explicit mechanistic/functional claims from scientific text.

INPUT
You will receive a list of sentences with integer IDs (sid). Each sentence is the only allowed source of truth.

TASK
Extract ONLY sentences/phrases that explicitly state an observed outcome/change related to a protein or molecular function, including:
- activity or signaling changes (only if explicitly stated)
- processing/maturation/cleavage changes
- secretion/trafficking/localization changes
- binding/interaction/complex assembly changes
- stability/abundance/levels/expression changes
- explicitly stated no-change/unchanged findings (only if explicitly stated)

DO NOT EXTRACT
- aims/objectives/plans (e.g., “we assessed”, “we investigated”, “we determined”)
- clinical association or prevalence statements
- mutation discovery statements without a protein-level outcome
- methods/assays/figures

STRICT NO-INFERENCE
- Do NOT infer. Do NOT use background knowledge.
- Do NOT convert “measured/assessed/investigated” into an outcome.

EVIDENCE REQUIREMENT
- Every claim must cite 1–3 evidence_sids.
- claim_text must be copied verbatim from one evidence sentence when possible, or a very tight paraphrase that does not add new information.
- All entities listed must appear in the evidence sentences.

OUTPUT JSON ONLY
Return JSON exactly in this format:
{
  "claims":[
    {
      "claim_id":"C0001",
      "evidence_sids":[35],
      "claim_text":"...",
      "entities":["..."],
      "scope_hint":"..."
    }
  ]
}

FIELD RULES
- claim_id: unique within the document.
- evidence_sids: integers only.
- claim_text: 1–2 sentences max.
- entities: 1–6 strings; include gene/protein names or key biological terms that appear in evidence.
- scope_hint: short phrase from evidence describing which variants/subjects/proteins the claim pertains to (e.g., “p.Ala347Val”, “these mutations”, “missense proteins”, “mutation carriers”).
"""


def build_numbered_sentence_prompt(sentences: List[Sent], max_chars: int) -> str:
    """
    Builds numbered sentence block like:
    [SENTENCE 0] ...
    [SENTENCE 1] ...
    ...
    Truncates by characters.
    """
    out = []
    total = 0
    for s in sentences:
        line = f"[SENTENCE {s.sid}] {s.text}\n"
        if total + len(line) > max_chars:
            break
        out.append(line)
        total += len(line)
    return "".join(out).strip()


def safe_json_loads_maybe(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempts to parse JSON even if model wraps it in markdown fences or extra text.
    """
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        pass

    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# =========================
# OpenRouter async client
# =========================

def require_api_key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        raise RuntimeError("Missing OPENROUTER_API_KEY env var (do not hardcode keys).")
    return k


async def call_openrouter(session: aiohttp.ClientSession, system_prompt: str, user_prompt: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/yourusername/yourproject",
        "X-Title": "PubMed Claim Extractor",
        "Content-Type": "application/json",
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": OUTPUT_MAX_TOKENS,  # verify this is a positive int
    }

    async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
        raw_text = await resp.text()

        if resp.status != 200:
            raise RuntimeError(f"OpenRouter {resp.status}: {raw_text[:4000]}")

        # Try JSON parse; if it fails, surface the raw body
        try:
            data = json.loads(raw_text)
        except Exception:
            raise RuntimeError(f"OpenRouter returned non-JSON body: {raw_text[:4000]}")

        # Defensive extraction
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices. Body: {raw_text[:4000]}")

        msg = (choices[0].get("message") or {})
        content = msg.get("content")

        # If content is empty, check common alternate fields and raise with context
        if not content or not str(content).strip():
            refusal = msg.get("refusal")
            tool_calls = msg.get("tool_calls")
            finish_reason = choices[0].get("finish_reason")

            raise RuntimeError(
                "Empty message.content from model.\n"
                f"finish_reason={finish_reason}\n"
                f"refusal={repr(refusal)[:500]}\n"
                f"tool_calls_present={tool_calls is not None}\n"
                f"body_preview={raw_text[:2000]}"
            )

        return str(content).strip()


# =========================
# Dataset loading (adapted to unknown schema)
# =========================

def load_dataset(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def iter_pubmed_records(data: Any):
    """
    Supports either:
    - dict: pmid -> entry
    - list: list of entries that contain pubmed_id/pmid
    """
    if isinstance(data, dict):
        for pmid, entry in data.items():
            yield str(pmid), entry
    elif isinstance(data, list):
        for entry in data:
            pmid = entry.get("pubmed_id") or entry.get("pmid") or entry.get("PMID")
            if pmid is None:
                continue
            yield str(pmid), entry
    else:
        raise ValueError("Unsupported dataset root type")


def extract_article_text(entry: Any) -> str:
    """
    Adjust here if your schema differs.
    Your earlier code used entry.get("article", "").
    """
    if isinstance(entry, dict):
        txt = entry.get("article") or entry.get("article_text") or entry.get("full_text") or entry.get("text") or ""
        return txt if isinstance(txt, str) else ""
    return ""


# =========================
# JSONL helpers / resume
# =========================

def load_done(out_path: str) -> set:
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = obj.get("pubmed_id") or obj.get("doc_id")
                if pid:
                    done.add(str(pid).replace("PMID:", ""))
            except Exception:
                continue
    return done


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def sentence_split_newline_fallback(text: str) -> List[Sent]:
    parts = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    sents = []
    cursor = 0
    for sid, p in enumerate(parts):
        idx = text.find(p, cursor)
        if idx == -1:
            idx = cursor
        sents.append(Sent(sid=sid, start_char=idx, end_char=idx+len(p), text=p))
        cursor = idx + len(p)
    return sents

def coerce_sid(x):
    """Normalize evidence_sids entries that may come back as ints or digit-strings."""
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        t = x.strip()
        if t.isdigit():
            return int(t)
    return None

# =========================
# Core per-PMID processing
# =========================

async def process_one_pmid(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    api_key: str,
    pmid: str,
    entry: Any,
) -> None:
    async with sem:
        raw = extract_article_text(entry)
        if not raw or not raw.strip() or raw.strip().lower() == "none":
            append_jsonl(ERROR_JSONL, {"pubmed_id": pmid, "status": "skipped", "reason": "no_article_text"})
            return

        if len(raw) > MAX_RAW_CHARS:
            raw = raw[:MAX_RAW_CHARS]

        # --- Section filtering (best-effort) ---
        filtered, did_split = filter_article_sections(raw)

        # Choose the universe text for segmentation/windowing
        if did_split and len(filtered) >= 3000 and len(filtered) >= 0.3 * len(raw):
            text_for_segmentation = filtered
            filter_mode = "sections"
        else:
            text_for_segmentation = raw
            did_split = False
            filter_mode = "raw"

        # Final guard
        if not text_for_segmentation.strip():
            append_jsonl(ERROR_JSONL, {"pubmed_id": pmid, "status": "skipped", "reason": "text_empty"})
            return

        # --- Sentence segmentation + robust fallback ---
        sents_all = segment_sentences(text_for_segmentation)
        if len(sents_all) < 3:
            sents_all = sentence_split_newline_fallback(text_for_segmentation)

        if len(sents_all) < 3:
            append_jsonl(ERROR_JSONL, {
                "pubmed_id": pmid,
                "status": "skipped",
                "reason": "too_few_sentences",
                "filter_mode": filter_mode,
                "raw_char_len": len(raw),
                "filtered_char_len": len(filtered),
                "text_for_segmentation_char_len": len(text_for_segmentation),
                "preview": text_for_segmentation[:400],
            })
            return

        # --- Select ±CONTEXT_WINDOW sentence windows around mechanistic keywords ---
        sents_for_prompt = select_sentence_windows(sents_all, MECH_KEYWORDS, window=CONTEXT_WINDOW)
        if not sents_for_prompt:
            sents_for_prompt = sents_all

        user_prompt = build_numbered_sentence_prompt(sents_for_prompt, PROMPT_MAX_CHARS)

        # --- LLM call ---
        try:
            resp_text = await call_openrouter(
                session=session,
                system_prompt=STAGE1_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                api_key=api_key,
            )
        except Exception as e:
            append_jsonl(ERROR_JSONL, {"pubmed_id": pmid, "status": "error", "reason": "llm_call", "detail": str(e)})
            return

        parsed = safe_json_loads_maybe(resp_text)
        if not parsed or "claims" not in parsed or not isinstance(parsed["claims"], list):
            append_jsonl(ERROR_JSONL, {
                "pubmed_id": pmid,
                "status": "error",
                "reason": "bad_json",
                "raw_model_output": resp_text[:2000],
            })
            return

        # --- Resolve evidence sentence IDs to spans/text ---
        sid_map = {s.sid: s for s in sents_all}
        resolved_claims = []

        for i, c in enumerate(parsed["claims"]):
            if not isinstance(c, dict):
                continue

            claim_id = c.get("claim_id") or f"C{i+1:04d}"

            evidence_sids = c.get("evidence_sids") or []
            if not isinstance(evidence_sids, list):
                evidence_sids = []

            evidence = []
            for sid_raw in evidence_sids:
                sid = coerce_sid(sid_raw)
                if sid is not None and sid in sid_map:
                    s = sid_map[sid]
                    evidence.append({
                        "sid": s.sid,
                        "start_char": s.start_char,
                        "end_char": s.end_char,
                        "text": s.text,
                    })

            c_out = dict(c)
            c_out["claim_id"] = claim_id
            c_out["evidence"] = evidence
            resolved_claims.append(c_out)

        append_jsonl(OUT_JSONL, {
            "pubmed_id": pmid,
            "doc_id": f"PMID:{pmid}",
            "status": "ok",
            "did_split": did_split,
            "filter_mode": filter_mode,
            "raw_char_len": len(raw),
            "filtered_char_len": len(filtered),
            "text_for_segmentation_char_len": len(text_for_segmentation),
            "num_sentences_total": len(sents_all),
            "num_sentences_used_in_prompt": len(sents_for_prompt),
            "model": LLM_MODEL,
            "claims": resolved_claims,
        })


# =========================
# Main
# =========================

async def main():
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    data = load_dataset(DATA_PATH)
    done = load_done(OUT_JSONL)

    sem = asyncio.Semaphore(CONCURRENCY)

    timeout = aiohttp.ClientTimeout(total=600, connect=30, sock_read=600)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, limit_per_host=CONCURRENCY)

    tasks = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for pmid, entry in iter_pubmed_records(data):
            if TEST_PMID and pmid != TEST_PMID:
                continue
            # if pmid in done:
            #     continue
            # tasks.append(process_one_pmid(sem, session, api_key, pmid, entry))
            tasks.append(
                asyncio.create_task(process_one_pmid(sem, session, api_key, pmid, entry))
            )

        # Stream completion
        for fut in asyncio.as_completed(tasks):
            await fut
            # tiny delay to be polite; tune/remove as needed
            await asyncio.sleep(0.01)

    print(f"Done. Wrote results to {OUT_JSONL} and errors to {ERROR_JSONL}.")

if __name__ == "__main__":
    asyncio.run(main())
