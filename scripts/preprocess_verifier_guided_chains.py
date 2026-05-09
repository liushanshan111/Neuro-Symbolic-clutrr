# scripts/preprocess_verifier_guided_chains.py
"""
Verifier-guided text-to-chain preprocessing.

This is the implementation for "方案2":
  LLM/candidate extraction -> symbolic verifier -> repaired relation chain.

Usage examples:
  # Local candidate extraction + verifier
  python3 scripts/preprocess_verifier_guided_chains.py \
      --input data/data_f70b574f/1.2,1.3,1.4_train_text.csv \
      --output data/data_f70b574f/1.2,1.3,1.4_train_vg_text.csv

  # Create JSONL prompts for an external LLM
  python3 scripts/preprocess_verifier_guided_chains.py \
      --input data/data_f70b574f/1.2,1.3,1.4_train_text.csv \
      --output data/data_f70b574f/1.2,1.3,1.4_train_vg_text.csv \
      --write-prompts outputs_advanced_fixed/vg_prompts_train.jsonl

  # Use external LLM JSONL outputs with fields: id, response
  python3 scripts/preprocess_verifier_guided_chains.py \
      --input data/data_f70b574f/1.2,1.3,1.4_train_text.csv \
      --output data/data_f70b574f/1.2,1.3,1.4_train_vg_text.csv \
      --llm-jsonl outputs_advanced_fixed/vg_llm_outputs_train.jsonl
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.verifier_guided_extractor import (  # noqa: E402
    BracketStoryCandidateExtractor,
    SymbolicChainVerifier,
    make_llm_prompt,
    parse_edge_types,
    parse_query,
    triples_from_llm_json,
)


def read_csv_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def write_csv_rows(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_llm_jsonl(path):
    if not path:
        return {}

    outputs = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            row_id = str(item.get("id", ""))
            response = item.get("response", item.get("output", ""))
            outputs[row_id] = response
    return outputs


def write_prompts(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            query = parse_query(row.get("query", ""))
            item = {
                "id": str(row.get("id", row.get("Unnamed: 0", ""))),
                "prompt": make_llm_prompt(row.get("story", ""), query),
            }
            f.write(json.dumps(item) + "\n")


def select_candidate_triples(row, llm_outputs, local_extractor):
    row_id = str(row.get("id", row.get("Unnamed: 0", "")))
    if row_id in llm_outputs:
        try:
            return triples_from_llm_json(llm_outputs[row_id]), "llm"
        except Exception as exc:
            return [], f"llm_parse_error:{exc}"

    return local_extractor.extract_triples(row.get("story", "")), "local"


def process(input_path, output_path, llm_jsonl=None, write_prompts_path=None, fallback=True):
    rows, fieldnames = read_csv_rows(input_path)
    if write_prompts_path:
        write_prompts(write_prompts_path, rows)
        print(f"LLM prompts written to: {write_prompts_path}")

    llm_outputs = load_llm_jsonl(llm_jsonl)
    local_extractor = BracketStoryCandidateExtractor()
    verifier = SymbolicChainVerifier()

    counters = defaultdict(int)
    output_rows = []

    extra_fields = [
        "vg_source",
        "vg_status",
        "vg_path",
        "vg_triples",
        "vg_feedback",
    ]
    output_fieldnames = list(fieldnames)
    for field in extra_fields:
        if field not in output_fieldnames:
            output_fieldnames.append(field)

    for row in rows:
        story = row.get("story", "")
        query = parse_query(row.get("query", ""))
        original_edges = parse_edge_types(row.get("edge_types", "[]"))
        target_length = len(original_edges) if original_edges else None

        triples, source = select_candidate_triples(row, llm_outputs, local_extractor)
        result = verifier.verify(story, query, triples, target_length=target_length)

        if result.status == "ok" and result.chain:
            row["edge_types"] = str(result.chain)
            row["extraction_source"] = "verifier_guided"
            counters["verified"] += 1
        elif fallback and original_edges:
            row["edge_types"] = str(original_edges)
            row["extraction_source"] = "fallback"
            counters[f"fallback_{result.status}"] += 1
        else:
            row["extraction_source"] = "dropped"
            counters[f"dropped_{result.status}"] += 1

        row["vg_source"] = source
        row["vg_status"] = result.status
        row["vg_path"] = json.dumps(result.path)
        row["vg_triples"] = json.dumps(result.triples)
        row["vg_feedback"] = json.dumps(result.feedback)
        output_rows.append(row)

    kept_rows = [row for row in output_rows if row.get("extraction_source") != "dropped"]
    write_csv_rows(output_path, kept_rows, output_fieldnames)

    total = len(rows)
    print("\n" + "=" * 70)
    print("Verifier-guided preprocessing complete")
    print("=" * 70)
    print(f"Input rows:  {total}")
    print(f"Output rows: {len(kept_rows)}")
    print(f"Verified:    {counters['verified']} ({counters['verified'] / max(total, 1):.1%})")
    for key in sorted(counters):
        if key != "verified":
            print(f"{key}: {counters[key]}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Verifier-guided text-to-chain preprocessing")
    parser.add_argument("--input", required=True, help="input CSV")
    parser.add_argument("--output", required=True, help="output CSV")
    parser.add_argument("--llm-jsonl", default=None, help="external LLM outputs JSONL: {id,response}")
    parser.add_argument("--write-prompts", default=None, help="write JSONL prompts for external LLM")
    parser.add_argument("--no-fallback", action="store_true", help="drop rows instead of using original edge_types")
    args = parser.parse_args()

    process(
        input_path=args.input,
        output_path=args.output,
        llm_jsonl=args.llm_jsonl,
        write_prompts_path=args.write_prompts,
        fallback=not args.no_fallback,
    )


if __name__ == "__main__":
    main()
