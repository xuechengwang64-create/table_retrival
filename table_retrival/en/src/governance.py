#!/usr/bin/env python3
"""
Table Retrieval — Governance Pipeline (English version).

Usage:
  python governance.py --dry-run           # print prompts without calling API
  python governance.py --batch-prompt1     # real API call
"""

import argparse
import json
import os
import sys

EN_SRC = os.path.dirname(os.path.abspath(__file__))
EN_ROOT = os.path.dirname(EN_SRC)
PROJECT_ROOT = os.path.dirname(EN_ROOT)
# en/src/ first → picks up English config.py
# parent src/ second → picks up shared data_loader, statistics, llm_client
sys.path.insert(0, EN_SRC)
sys.path.insert(1, os.path.join(PROJECT_ROOT, "src"))

from config import PROMPT_TABLE_TO_KEYWORDS, PROMPT_KEYWORDS_TO_TABLES
from data_loader import load_table_metadata, load_query_samples
from llm_client import LLMClient
from statistics import (
    compute_usage_stats,
    format_all_metadata_for_prompt2,
    format_all_stats_for_prompt2,
    format_table_stats_for_prompt1,
)


def build_prompt1_batch(tables, stats) -> str:
    """Build a single combined Prompt-1 user message for all tables."""
    parts = []
    for tbl, meta in tables.items():
        stats_str = format_table_stats_for_prompt1(stats, tbl)
        parts.append(f"""---
## Metadata
Table name: {meta.table_name}
Business name: {meta.business_name_cn}
Description: {meta.description_cn}
High-frequency columns:
{meta.columns_info_str}

## Query Statistics
{stats_str}""")
    return "\n\n".join(parts)


def run_prompt1(client: LLMClient, tables, stats) -> dict:
    """Run Prompt 1 in a single batch call."""
    user_msg = build_prompt1_batch(tables, stats)
    if client.dry_run:
        print(f"\n{'─' * 60}")
        print("Prompt 1 — BATCH (all tables)")
        print(f"{'─' * 60}")
    raw = client.chat(PROMPT_TABLE_TO_KEYWORDS, user_msg)
    try:
        return client.extract_json(raw)
    except json.JSONDecodeError:
        print("[WARN] Failed to parse Prompt 1 JSON. Saving raw output.")
        os.makedirs(os.path.join(PROJECT_ROOT, "output"), exist_ok=True)
        with open(os.path.join(PROJECT_ROOT, "output/prompt1_debug.txt"), "w") as f:
            f.write(raw)
        return {}


def run_prompt2(client: LLMClient, tables, stats) -> dict:
    """Run Prompt 2: keyword-combinations → table-sets."""
    metadata_str = format_all_metadata_for_prompt2(tables)
    stats_str = format_all_stats_for_prompt2(stats)
    user_msg = f"""## All Table Metadata
{metadata_str}
## All Table Query Statistics
{stats_str}"""

    if client.dry_run:
        print(f"\n{'─' * 60}")
        print("Prompt 2 — Keyword Combinations → Tables")
        print(f"{'─' * 60}")

    raw = client.chat(PROMPT_KEYWORDS_TO_TABLES, user_msg)
    try:
        return client.extract_json(raw)
    except json.JSONDecodeError:
        print("[WARN] Failed to parse Prompt 2 JSON. Saving raw output.")
        os.makedirs(os.path.join(PROJECT_ROOT, "output"), exist_ok=True)
        with open(os.path.join(PROJECT_ROOT, "output/prompt2_debug.txt"), "w") as f:
            f.write(raw)
        return {}


def save_output(data: dict, filename: str, output_dir: str):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved → {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Table Retrieval Governance (EN)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default=os.path.join(EN_ROOT, "output"))
    parser.add_argument("--metadata-dir", default=os.path.join(EN_ROOT, "sample_data/metadata"))
    parser.add_argument("--query-csv", default=os.path.join(EN_ROOT, "sample_data/query_samples.csv"))
    args = parser.parse_args()

    print("[1/5] Loading table metadata...")
    tables = load_table_metadata(args.metadata_dir)
    print(f"  → Loaded {len(tables)} tables")

    print("[2/5] Loading query samples...")
    samples = load_query_samples(args.query_csv)
    print(f"  → Loaded {len(samples)} samples")

    print("[3/5] Computing usage statistics...")
    stats = compute_usage_stats(samples, tables)
    for tbl, s in stats.items():
        print(f"  {tbl}: usage={s['usage_count']}, co_tables={list(s['co_tables'].keys())}")

    client = LLMClient(model=args.model, dry_run=args.dry_run)

    print("[4/5] Running Prompt 1: table → keywords ...")
    keywords_map = run_prompt1(client, tables, stats)
    print(f"  → Got keywords for {len(keywords_map)} tables")

    print("[5/5] Running Prompt 2: keyword combinations → tables ...")
    inverted_map = run_prompt2(client, tables, stats)
    print(f"  → Got {len(inverted_map)} keyword-combination mappings")

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.dry_run:
        save_output(keywords_map, "table_keywords.json", args.output_dir)
        save_output(inverted_map, "keyword_to_tables.json", args.output_dir)
        save_output(stats, "co_tables.json", args.output_dir)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tables indexed:     {len(keywords_map)}")
    print(f"Keyword mappings:   {len(inverted_map)}")
    for k, v in inverted_map.items():
        print(f"  {k} → {v}")
    if args.dry_run:
        print("[NOTE] --dry-run used; output files contain empty JSON.")


if __name__ == "__main__":
    main()
