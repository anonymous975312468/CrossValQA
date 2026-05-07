#!/usr/bin/env python3
"""
Smoke test different models for Phase 2 - Compare outputs with context
"""

import json
import asyncio
import aiohttp
import os
import time
import sys
import re
import random

sys.path.insert(0, "./claim_extractor/phase_0")
from prefilter import split_into_sentences

INPUT_PHASE1 = "./claim_extractor/phase_1/output/phase1_qa_pairs.jsonl"
INPUT_ARTICLES = "./database/articles_with_mutations.json"

SYSTEM_PROMPT = """You are a scientific QA system specializing in genetics and molecular biology.

Your task is to answer a question about a genetic mutation using ONLY the provided context sentences from a research paper.

CRITICAL RULES:
1. Answer ONLY based on what is explicitly stated in the context
2. Include the mutation identifier in your answer
3. If the context does not contain enough information to answer the question, say so clearly
4. Provide a rationale that:
   - Quotes relevant phrases from the context (in quotation marks)
   - Cites the specific sentence number(s) in brackets [N]
   - Explains how the evidence supports your answer
5. Do NOT use any external knowledge about the gene or mutation
6. Do NOT assume or infer information not directly stated

OUTPUT FORMAT (JSON only):
{
  "answer": "Your answer here, including the mutation identifier",
  "rationale": "Quote from context [sentence #]. Explanation of how this supports the answer.",
  "evidence_sentence_ids": [list of sentence numbers used],
  "confidence": "high|medium|low",
  "answerable": true
}

If the question cannot be answered from the context, respond with:
{
  "answer": "The provided context does not contain sufficient information to answer this question about [mutation].",
  "rationale": "Explanation of what information is missing.",
  "evidence_sentence_ids": [],
  "confidence": "low",
  "answerable": false
}"""

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


def find_mutation_sentences(sentences, mutation):
    indices = set()
    escaped = re.escape(mutation)
    for i, sent in enumerate(sentences):
        if re.search(escaped, sent, re.IGNORECASE):
            indices.add(i)
    return indices


def build_context(sentences, mutation, window=5):
    indices = find_mutation_sentences(sentences, mutation)
    if not indices:
        return "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences[:20])), set()
    
    context_indices = set()
    for idx in indices:
        start = max(0, idx - window)
        end = min(len(sentences), idx + window + 1)
        context_indices.update(range(start, end))
    
    context_text = "\n".join(f"[{i}] {sentences[i]}" for i in sorted(context_indices))
    return context_text, indices


async def test_model(session, api_key, model, prompt, system_prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1000,
        "temperature": 0.3,
    }
    
    start = time.time()
    try:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            elapsed = time.time() - start
            if resp.status == 200:
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return {
                    "success": True,
                    "content": content,
                    "elapsed": elapsed,
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                }
            else:
                text = await resp.text()
                return {"success": False, "error": f"HTTP {resp.status}: {text[:200]}", "elapsed": elapsed}
    except Exception as e:
        return {"success": False, "error": str(e), "elapsed": time.time() - start}


async def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set")
        return
    
    # Load data
    print("Loading data...")
    with open(INPUT_ARTICLES) as f:
        articles = json.load(f)
    
    # Load all Phase 1 entries
    print("Loading Phase 1 results...")
    all_entries = []
    with open(INPUT_PHASE1) as f:
        for line in f:
            entry = json.loads(line)
            if entry["pmid"] in articles:
                all_entries.append(entry)
    
    print(f"Loaded {len(all_entries):,} entries")
    
    # Random sample of 3
    random.seed()  # True random each run
    sampled_entries = random.sample(all_entries, 3)
    
    # Build samples
    samples = []
    for entry in sampled_entries:
        pmid = entry["pmid"]
        sentences = split_into_sentences(articles[pmid].get("article", ""))
        mutation = entry["mutation"]
        
        qa = entry.get("qa_pairs", [])[0]  # First QA
        context, mutation_indices = build_context(sentences, mutation)
        prompt = USER_PROMPT_TEMPLATE.format(
            mutation=mutation,
            question=qa["question"],
            context=context
        )
        samples.append({
            "pmid": pmid,
            "mutation": mutation,
            "question": qa["question"],
            "context": context,
            "mutation_indices": mutation_indices,
            "prompt": prompt,
        })
    
    print(f"Testing with {len(samples)} randomly selected samples\n")
    
    # Models to test
    models = [
        "deepseek/deepseek-chat",
        "anthropic/claude-3-haiku",
        "meta-llama/llama-3.1-8b-instruct",
    ]
    
    async with aiohttp.ClientSession() as session:
        for i, sample in enumerate(samples):
            print("=" * 80)
            print(f"SAMPLE {i+1}")
            print("=" * 80)
            print(f"PMID: {sample['pmid']}")
            print(f"Mutation: {sample['mutation']}")
            print(f"Question: {sample['question']}")
            print(f"Mutation found in sentences: {sample['mutation_indices']}")
            print(f"\nCONTEXT (±5 sentences around mutation mentions):")
            print("-" * 40)
            print(sample['context'])
            print("-" * 40)
            
            for model in models:
                print(f"\n{'='*60}")
                print(f"MODEL: {model}")
                print(f"{'='*60}")
                
                result = await test_model(session, api_key, model, sample["prompt"], SYSTEM_PROMPT)
                
                if result["success"]:
                    print(f"Latency: {result['elapsed']:.2f}s | Tokens: {result['input_tokens']} in, {result['output_tokens']} out")
                    print(f"\nResponse:")
                    print(result['content'])
                else:
                    print(f"FAILED: {result['error']}")
            
            print("\n\n")


if __name__ == "__main__":
    asyncio.run(main())