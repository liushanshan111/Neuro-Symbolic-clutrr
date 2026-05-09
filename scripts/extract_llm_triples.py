"""
Extract CLUTRR family-relation triples from text with an LLM.

This script calls an OpenAI-compatible Chat Completions API and writes JSONL
that can be consumed by scripts/preprocess_verifier_guided_chains.py:

  {"id": "...", "response": "{\"triples\":[...]}", "triples": [...]}

Example:
  export LLM_API_KEY=...
  python3 scripts/extract_llm_triples.py \
      --input data/data_f70b574f/1.10_test_text.csv \
      --output outputs_advanced_fixed/llm_triples_1.10.jsonl \
      --api-url https://openrouter.ai/api/v1/chat/completions \
      --model meta-llama/llama-3.3-70b-instruct

Then verify and convert triples to edge_types:
  python3 scripts/preprocess_verifier_guided_chains.py \
      --input data/data_f70b574f/1.10_test_text.csv \
      --output data/data_f70b574f/1.10_test_llm_vg_text.csv \
      --llm-jsonl outputs_advanced_fixed/llm_triples_1.10.jsonl
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.verifier_guided_extractor import (  # noqa: E402
    make_llm_prompt,
    parse_query,
    triples_from_llm_json,
)


DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = (
    "You extract family relation triples from short CLUTRR stories. "
    "Return strict JSON only. Do not explain."
)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_done_ids(path):
    done = set()
    if not path or not Path(path).exists():
        return done

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line).get("id", "")))
            except json.JSONDecodeError:
                continue
    return done


def row_id(row):
    return str(row.get("id", row.get("Unnamed: 0", "")))


def build_messages(row):
    query = parse_query(row.get("query", ""))
    prompt = make_llm_prompt(row.get("story", ""), query)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def parse_chat_response(payload):
    choices = payload.get("choices", [])
    if not choices:
        raise ValueError("missing choices in API response")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                text_parts.append(str(part.get("text", "")))
            else:
                text_parts.append(str(part))
        content = "".join(text_parts)
    return str(content).strip()


def call_llm(args, messages):
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.response_format_json:
        body["response_format"] = {"type": "json_object"}

    last_error = None
    for attempt in range(args.max_retries):
        try:
            resp = requests.post(args.api_url, headers=headers, json=body, timeout=args.timeout)
            if resp.status_code == 429:
                wait = min(args.max_retry_wait, 2 ** (attempt + 2))
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return parse_chat_response(resp.json())
        except Exception as exc:
            last_error = exc
            if attempt + 1 < args.max_retries:
                time.sleep(min(args.max_retry_wait, 2 ** attempt))

    raise RuntimeError(f"LLM request failed after {args.max_retries} attempts: {last_error}")


def write_jsonl_item(handle, item):
    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    handle.flush()


def main():
    parser = argparse.ArgumentParser(description="Extract relation triples from CLUTRR text with an LLM")
    parser.add_argument("--input", required=True, help="input CLUTRR text CSV")
    parser.add_argument("--output", required=True, help="output JSONL path")
    parser.add_argument("--api-url", default=os.environ.get("LLM_API_URL", DEFAULT_API_URL))
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct"))
    parser.add_argument("--limit", type=int, default=None, help="extract only first N unfinished rows")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep after each successful request")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-retry-wait", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--response-format-json", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip ids already present in --output")
    parser.add_argument("--dry-run", action="store_true", help="print the first prompt without calling the API")
    args = parser.parse_args()

    rows = read_rows(args.input)
    if args.dry_run:
        if not rows:
            raise RuntimeError("input CSV has no rows")
        print(json.dumps(build_messages(rows[0]), ensure_ascii=False, indent=2))
        return

    if not args.api_key:
        raise RuntimeError("Set --api-key or LLM_API_KEY/OPENROUTER_API_KEY")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output_path) if args.resume else set()

    todo = [row for row in rows if row_id(row) not in done_ids]
    if args.limit is not None:
        todo = todo[: args.limit]

    mode = "a" if args.resume else "w"
    ok = 0
    failed = 0

    with open(output_path, mode) as f:
        for row in tqdm(todo, desc="LLM triple extraction"):
            rid = row_id(row)
            try:
                response = call_llm(args, build_messages(row))
                try:
                    triples = triples_from_llm_json(response)
                    parse_error = None
                except Exception as exc:
                    triples = []
                    parse_error = str(exc)

                write_jsonl_item(
                    f,
                    {
                        "id": rid,
                        "response": response,
                        "triples": triples,
                        "parse_error": parse_error,
                    },
                )
                ok += 1
                if args.sleep > 0:
                    time.sleep(args.sleep)
            except Exception as exc:
                failed += 1
                write_jsonl_item(
                    f,
                    {
                        "id": rid,
                        "response": "",
                        "triples": [],
                        "error": str(exc),
                    },
                )

    print("\n" + "=" * 70)
    print("LLM triple extraction complete")
    print("=" * 70)
    print(f"Input rows: {len(rows)}")
    print(f"Requested:  {len(todo)}")
    print(f"Success:    {ok}")
    print(f"Failed:     {failed}")
    print(f"Output:     {output_path}")


if __name__ == "__main__":
    main()
