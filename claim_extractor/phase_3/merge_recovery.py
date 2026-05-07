#!/usr/bin/env python3
"""
Merge recovered grounded items into the main Phase 3 output.
"""

import json

MAIN_FILE = "./claim_extractor/phase_3/output/phase3_validated.jsonl"
RECOVERY_FILE = "./claim_extractor/phase_3/output/recovery/phase3_recovered.jsonl"
OUTPUT_FILE = "./claim_extractor/phase_3/output/phase3_validated_merged.jsonl"

def main():
    # Load main output
    print("Loading main Phase 3 output...")
    main_items = []
    main_keys = set()

    with open(MAIN_FILE) as f:
        for line in f:
            item = json.loads(line)
            main_items.append(item)
            key = f"{item['pmid']}|{item['mutation']}|{item['question'][:80]}"
            main_keys.add(key)

    print(f"  Main items: {len(main_items):,}")

    # Load recovery items (only grounded ones)
    print("Loading recovery results...")
    recovered_grounded = []
    recovered_ungrounded = 0
    duplicates = 0

    with open(RECOVERY_FILE) as f:
        for line in f:
            item = json.loads(line)
            grounding = item.get("grounding", "")

            if grounding in ("cross-grounded", "grounded"):
                key = f"{item['pmid']}|{item['mutation']}|{item['question'][:80]}"
                if key not in main_keys:
                    recovered_grounded.append(item)
                    main_keys.add(key)
                else:
                    duplicates += 1
            else:
                recovered_ungrounded += 1

    print(f"  Recovered grounded: {len(recovered_grounded):,}")
    print(f"  Recovered ungrounded (skipped): {recovered_ungrounded:,}")
    print(f"  Duplicates (skipped): {duplicates:,}")

    # Merge
    merged = main_items + recovered_grounded

    # Write output
    print(f"\nWriting merged output...")
    with open(OUTPUT_FILE, "w") as f:
        for item in merged:
            f.write(json.dumps(item) + "\n")

    # Final stats
    from collections import Counter
    counts = Counter(item.get("grounding") for item in merged)

    print(f"\n=== MERGED RESULTS ===")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total items: {len(merged):,}")
    print(f"  Cross-grounded: {counts['cross-grounded']:,}")
    print(f"  Grounded: {counts['grounded']:,}")
    print(f"  Ungrounded: {counts['ungrounded']:,}")
    print(f"\nTotal VALID (cross-grounded + grounded): {counts['cross-grounded'] + counts['grounded']:,}")

if __name__ == "__main__":
    main()
