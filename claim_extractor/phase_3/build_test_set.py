
#!/usr/bin/env python3
"""Build a diverse test set for judge model comparison."""

import json
import random

INPUT = "./claim_extractor/phase_2/output/phase2_qa_pairs.jsonl"
OUTPUT_DIR = "./claim_extractor/phase_3/smoke_tests"

# Load all data
all_items = []
with open(INPUT) as f:
    for line in f:
        all_items.append(json.loads(line))

print(f"Loaded {len(all_items):,} items")

# Categorize
same_evidence = []  # Both cite same sentences
diff_evidence = []  # Cite different sentences
no_p2_evidence = [] # Phase 2 found no evidence

for item in all_items:
    # Flatten in case of nested lists
    p1_raw = item['phase1_evidence_ids']
    p2_raw = item['phase2_evidence_ids']
    
    # Handle potential nested lists or non-list types
    if isinstance(p1_raw, list):
        p1 = set(x for x in p1_raw if not isinstance(x, list))
    else:
        p1 = set()
    
    if isinstance(p2_raw, list):
        p2 = set(x for x in p2_raw if not isinstance(x, list))
    else:
        p2 = set()
    
    if not p2:
        no_p2_evidence.append(item)
    elif p1 and p1.intersection(p2):
        same_evidence.append(item)
    else:
        diff_evidence.append(item)

print(f"Same evidence: {len(same_evidence):,}")
print(f"Different evidence: {len(diff_evidence):,}")
print(f"No P2 evidence: {len(no_p2_evidence):,}")

# Sample diverse test set
random.seed(42)  # Reproducible
test_set = []

# 20 same evidence (expect: supported)
test_set.extend(random.sample(same_evidence, min(20, len(same_evidence))))

# 20 different evidence (mixed expectation)
test_set.extend(random.sample(diff_evidence, min(20, len(diff_evidence))))

# 10 no P2 evidence (expect: not supported for R2->A1)
test_set.extend(random.sample(no_p2_evidence, min(10, len(no_p2_evidence))))

print(f"\nTest set: {len(test_set)} items")

# Save
with open(f"{OUTPUT_DIR}/test_set_50.jsonl", "w") as f:
    for item in test_set:
        f.write(json.dumps(item) + "\n")

print(f"Saved to {OUTPUT_DIR}/test_set_50.jsonl")
