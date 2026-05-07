#!/usr/bin/env python3
"""
Smoke test for Phase 3 judge models on 50 diverse cases.
Outputs results to file for later inspection.
"""

import json
import os
import asyncio
import aiohttp
import time
from typing import Dict, Tuple, Optional

API_KEY = os.environ.get("OPENROUTER_API_KEY")

MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-nemo",
    "openai/gpt-4o-mini",
]

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


async def call_model(session: aiohttp.ClientSession, model: str, mutation: str, question: str, answer: str, rationale: str) -> Tuple[Optional[bool], str, float]:
    """Call a model and return (supported, reasoning, latency)."""
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/mutation-qa-generator",
        "X-Title": "Mutation QA Judge Test",
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_PROMPT.format(
                mutation=mutation, question=question, answer=answer, rationale=rationale
            )},
        ],
        "max_tokens": 300,
        "temperature": 0.1,
    }
    
    start = time.time()
    try:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            latency = time.time() - start
            if resp.status == 200:
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                content = content.strip()
                if content.startswith("```"):
                    content = "\n".join(l for l in content.split("\n") if not l.startswith("```"))
                try:
                    parsed = json.loads(content)
                    return parsed.get("supported"), parsed.get("reasoning", ""), latency
                except:
                    return None, f"Parse error: {content[:100]}", latency
            else:
                text = await resp.text()
                return None, f"HTTP {resp.status}: {text[:200]}", latency
    except Exception as e:
        return None, str(e), time.time() - start


async def test_item(session: aiohttp.ClientSession, item: Dict, idx: int) -> Dict:
    """Test all models on a single item."""
    
    result = {
        "idx": idx,
        "pmid": item["pmid"],
        "mutation": item["mutation"],
        "question": item["question"],
        "phase1_answer": item["phase1_answer"],
        "phase2_answer": item["phase2_answer"],
        "phase1_rationale": item["phase1_rationale"],
        "phase2_rationale": item["phase2_rationale"],
        "phase1_evidence_ids": item["phase1_evidence_ids"],
        "phase2_evidence_ids": item["phase2_evidence_ids"],
        "models": {}
    }
    
    for model in MODELS:
        model_results = {}
        
        # R2 -> A1
        supported, reasoning, latency = await call_model(
            session, model,
            item["mutation"],
            item["question"],
            item["phase1_answer"],
            item["phase2_rationale"]
        )
        model_results["r2_a1"] = {
            "supported": supported,
            "reasoning": reasoning,
            "latency": latency
        }
        
        # R1 -> A2
        supported, reasoning, latency = await call_model(
            session, model,
            item["mutation"],
            item["question"],
            item["phase2_answer"],
            item["phase1_rationale"]
        )
        model_results["r1_a2"] = {
            "supported": supported,
            "reasoning": reasoning,
            "latency": latency
        }
        
        result["models"][model] = model_results
    
    return result


async def main():
    INPUT = "./claim_extractor/phase_3/smoke_tests/test_set_50.jsonl"
    OUTPUT = "./claim_extractor/phase_3/smoke_tests/results_50.jsonl"
    SUMMARY = "./claim_extractor/phase_3/smoke_tests/summary_50.txt"
    
    # Load test cases
    test_cases = []
    with open(INPUT) as f:
        for line in f:
            test_cases.append(json.loads(line))
    
    print(f"Testing {len(test_cases)} cases across {len(MODELS)} models")
    print(f"This will make {len(test_cases) * len(MODELS) * 2} API calls")
    
    results = []
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        for idx, item in enumerate(test_cases):
            result = await test_item(session, item, idx)
            results.append(result)
            
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {idx + 1}/{len(test_cases)} ({elapsed:.1f}s)")
            
            await asyncio.sleep(0.5)  # Rate limit buffer
    
    # Save detailed results
    with open(OUTPUT, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nDetailed results saved to {OUTPUT}")
    
    # Generate summary
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("PHASE 3 JUDGE MODEL COMPARISON - 50 CASES")
    summary_lines.append("=" * 80)
    
    # Aggregate stats per model
    for model in MODELS:
        r2_a1_true = sum(1 for r in results if r["models"][model]["r2_a1"]["supported"] is True)
        r2_a1_false = sum(1 for r in results if r["models"][model]["r2_a1"]["supported"] is False)
        r2_a1_none = sum(1 for r in results if r["models"][model]["r2_a1"]["supported"] is None)
        
        r1_a2_true = sum(1 for r in results if r["models"][model]["r1_a2"]["supported"] is True)
        r1_a2_false = sum(1 for r in results if r["models"][model]["r1_a2"]["supported"] is False)
        r1_a2_none = sum(1 for r in results if r["models"][model]["r1_a2"]["supported"] is None)
        
        avg_latency = sum(
            r["models"][model]["r2_a1"]["latency"] + r["models"][model]["r1_a2"]["latency"]
            for r in results
        ) / (len(results) * 2)
        
        summary_lines.append(f"\n{model}:")
        summary_lines.append(f"  R2->A1: True={r2_a1_true}, False={r2_a1_false}, Error={r2_a1_none}")
        summary_lines.append(f"  R1->A2: True={r1_a2_true}, False={r1_a2_false}, Error={r1_a2_none}")
        summary_lines.append(f"  Avg latency: {avg_latency:.2f}s")
    
    # Agreement analysis
    summary_lines.append("\n" + "=" * 80)
    summary_lines.append("MODEL AGREEMENT ANALYSIS")
    summary_lines.append("=" * 80)
    
    agreements_r2_a1 = 0
    agreements_r1_a2 = 0
    
    for r in results:
        votes_r2_a1 = [r["models"][m]["r2_a1"]["supported"] for m in MODELS]
        votes_r1_a2 = [r["models"][m]["r1_a2"]["supported"] for m in MODELS]
        
        valid_r2_a1 = [v for v in votes_r2_a1 if v is not None]
        valid_r1_a2 = [v for v in votes_r1_a2 if v is not None]
        
        if len(set(valid_r2_a1)) == 1 and len(valid_r2_a1) == 3:
            agreements_r2_a1 += 1
        if len(set(valid_r1_a2)) == 1 and len(valid_r1_a2) == 3:
            agreements_r1_a2 += 1
    
    summary_lines.append(f"R2->A1 unanimous agreement: {agreements_r2_a1}/{len(results)} ({100*agreements_r2_a1/len(results):.1f}%)")
    summary_lines.append(f"R1->A2 unanimous agreement: {agreements_r1_a2}/{len(results)} ({100*agreements_r1_a2/len(results):.1f}%)")
    
    # Find disagreements
    summary_lines.append("\n" + "=" * 80)
    summary_lines.append("DISAGREEMENT CASES (for manual review)")
    summary_lines.append("=" * 80)
    
    for r in results:
        votes_r2_a1 = [r["models"][m]["r2_a1"]["supported"] for m in MODELS]
        valid_r2_a1 = [v for v in votes_r2_a1 if v is not None]
        
        if len(set(valid_r2_a1)) > 1:
            summary_lines.append(f"\nCase {r['idx']} (PMID: {r['pmid']}, Mutation: {r['mutation']})")
            summary_lines.append(f"  Q: {r['question'][:60]}...")
            for m in MODELS:
                v = r["models"][m]["r2_a1"]["supported"]
                summary_lines.append(f"  {m}: {v}")
    
    summary = "\n".join(summary_lines)
    
    with open(SUMMARY, "w") as f:
        f.write(summary)
    
    print(summary)
    print(f"\nSummary saved to {SUMMARY}")


if __name__ == "__main__":
    asyncio.run(main())