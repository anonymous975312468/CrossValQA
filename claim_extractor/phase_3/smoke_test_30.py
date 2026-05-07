#!/usr/bin/env python3
import json, os, asyncio, aiohttp, time

API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODELS = ["meta-llama/llama-3.1-8b-instruct", "mistralai/mistral-nemo", "openai/gpt-4o-mini"]

JUDGE_SYSTEM_PROMPT = """You are evaluating whether a rationale supports an answer to a question about a genetic mutation.
Determine if the RATIONALE logically supports the ANSWER to the QUESTION.
Output JSON only: {"supported": true/false, "reasoning": "Brief explanation"}"""

JUDGE_USER_PROMPT = """MUTATION: {mutation}
QUESTION: {question}
ANSWER: {answer}
RATIONALE: {rationale}
Does this rationale support this answer? Output JSON only."""

async def call_model(session, model, mutation, question, answer, rationale):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "HTTP-Referer": "https://github.com/mutation-qa"}
    payload = {"model": model, "messages": [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": JUDGE_USER_PROMPT.format(mutation=mutation, question=question, answer=answer, rationale=rationale)}], "max_tokens": 300, "temperature": 0.1}
    start = time.time()
    try:
        async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            latency = time.time() - start
            if resp.status == 200:
                data = await resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                if content.startswith("```"): content = "\n".join(l for l in content.split("\n") if not l.startswith("```"))
                try:
                    parsed = json.loads(content)
                    return parsed.get("supported"), parsed.get("reasoning", ""), latency
                except: return None, f"Parse error", latency
            return None, f"HTTP {resp.status}", latency
    except Exception as e: return None, str(e), time.time() - start

async def main():
    results = []
    with open("./claim_extractor/phase_3/smoke_tests/test_set_30.jsonl") as f:
        test_cases = [json.loads(line) for line in f]
    print(f"Testing {len(test_cases)} cases across {len(MODELS)} models ({len(test_cases)*len(MODELS)*2} API calls)")
    
    async with aiohttp.ClientSession() as session:
        for idx, item in enumerate(test_cases):
            result = {"idx": idx, "pmid": item["pmid"], "mutation": item["mutation"], "question": item["question"], "phase1_answer": item["phase1_answer"], "phase2_answer": item["phase2_answer"], "phase1_rationale": item["phase1_rationale"], "phase2_rationale": item["phase2_rationale"], "models": {}}
            for model in MODELS:
                r2a1 = await call_model(session, model, item["mutation"], item["question"], item["phase1_answer"], item["phase2_rationale"])
                r1a2 = await call_model(session, model, item["mutation"], item["question"], item["phase2_answer"], item["phase1_rationale"])
                result["models"][model] = {"r2_a1": {"supported": r2a1[0], "reasoning": r2a1[1], "latency": r2a1[2]}, "r1_a2": {"supported": r1a2[0], "reasoning": r1a2[1], "latency": r1a2[2]}}
            results.append(result)
            # Save after each to avoid losing progress
            with open("./claim_extractor/phase_3/smoke_tests/results_30.jsonl", "w") as f:
                for r in results: f.write(json.dumps(r) + "\n")
            if (idx + 1) % 5 == 0: print(f"Progress: {idx + 1}/{len(test_cases)}")
    print("Done!")

asyncio.run(main())
