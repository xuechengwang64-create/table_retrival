#!/usr/bin/env python3
"""
Table Retrieval — Governance Pipeline Demo.

Two-stage prompt pipeline:
  Prompt 1: table_name → keywords        (forward index)
  Prompt 2: keyword combinations → tables (inverted index)

Usage:
  python main.py                          # real API call
  python main.py --dry-run                # print prompts without calling API
  python main.py --output-dir ./output    # specify output directory
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from config import PROMPT_TABLE_TO_KEYWORDS, PROMPT_KEYWORDS_TO_TABLES
from data_loader import load_table_metadata, load_query_samples
from llm_client import LLMClient
from statistics import (
    compute_usage_stats,
    format_all_metadata_for_prompt2,
    format_all_stats_for_prompt2,
    format_table_stats_for_prompt1,
)


def build_prompt1_messages(
    tables, stats
) -> list[dict]:
    """Build individual prompt1 calls — one per table."""
    messages = []
    for tbl, meta in tables.items():
        stats_str = format_table_stats_for_prompt1(stats, tbl)
        user_msg = f"""## 元数据
表名: {meta.table_name}
业务名称: {meta.business_name_cn}
描述: {meta.description_cn}
高频列信息:
{meta.columns_info_str}

## 查询统计
{stats_str}"""
        messages.append({"table_name": tbl, "user_message": user_msg})
    return messages


def run_prompt1(client: LLMClient, tables, stats) -> dict[str, list[str]]:
    """
    Run Prompt 1: get table → keywords mapping.

    Calls LLM once per table, then merges results into a single JSON.
    """
    all_keywords = {}
    per_table_msgs = build_prompt1_messages(tables, stats)

    for item in per_table_msgs:
        tbl = item["table_name"]
        user_msg = item["user_message"]
        # Using system prompt for role, user message for data
        system_prompt = PROMPT_TABLE_TO_KEYWORDS
        system_prompt = system_prompt.replace("{table_name}", tbl)

        if client.dry_run:
            print(f"\n{'─' * 60}")
            print(f"Prompt 1 — Table: {tbl}")
            print(f"{'─' * 60}")
        raw = client.chat(system_prompt, user_msg)

        try:
            result = client.extract_json(raw)
            # Result could be {table_name: [...]} or just {...}
            if tbl in result:
                all_keywords[tbl] = result[tbl]
            else:
                # Multi-table response — extract the one we want
                all_keywords.update(result)
        except json.JSONDecodeError:
            print(f"[WARN] Failed to parse JSON for table {tbl}. Raw response saved.")
            # Save raw for debugging
            os.makedirs("output", exist_ok=True)
            with open(f"output/prompt1_debug_{tbl}.txt", "w") as f:
                f.write(raw)

    return all_keywords


def run_prompt1_batched(client: LLMClient, tables, stats) -> dict[str, list[str]]:
    """Run Prompt 1 in a single batch call: all tables at once, output one merged JSON."""
    # Build a combined user message with all tables
    parts = []
    for tbl, meta in tables.items():
        stats_str = format_table_stats_for_prompt1(stats, tbl)
        parts.append(f"""---
## 元数据
表名: {meta.table_name}
业务名称: {meta.business_name_cn}
描述: {meta.description_cn}
高频列信息:
{meta.columns_info_str}

## 查询统计
{stats_str}""")

    full_user_msg = "\n\n".join(parts)

    if client.dry_run:
        print(f"\n{'─' * 60}")
        print("Prompt 1 — BATCH (all tables)")
        print(f"{'─' * 60}")

    raw = client.chat(PROMPT_TABLE_TO_KEYWORDS, full_user_msg)
    try:
        return client.extract_json(raw)
    except json.JSONDecodeError:
        print("[WARN] Failed to parse batched Prompt 1 JSON. Saving raw output.")
        os.makedirs("output", exist_ok=True)
        with open("output/prompt1_debug_batch.txt", "w") as f:
            f.write(raw)
        return {}


def run_prompt2(client: LLMClient, tables, stats) -> dict:
    """
    Run Prompt 2: get keyword-combinations → table-sets mapping.
    Single call with all table info.
    """
    metadata_str = format_all_metadata_for_prompt2(tables)
    stats_str = format_all_stats_for_prompt2(stats)

    user_msg = f"""## 所有表元数据
{metadata_str}
## 所有表查询统计
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
        os.makedirs("output", exist_ok=True)
        with open("output/prompt2_debug.txt", "w") as f:
            f.write(raw)
        return {}


def save_output(data: dict, filename: str, output_dir: str):
    """Save a dict as formatted JSON."""
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved → {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Table Retrieval Governance Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    parser.add_argument(
        "--batch-prompt1", action="store_true",
        help="Run Prompt 1 as a single batch call instead of per-table"
    )
    parser.add_argument(
        "--output-dir", default=os.path.join(PROJECT_ROOT, "output"),
        help="Output directory for JSON results"
    )
    parser.add_argument(
        "--metadata-dir", default=os.path.join(PROJECT_ROOT, "sample_data/metadata"),
        help="Path to metadata YAML directory"
    )
    parser.add_argument(
        "--query-csv", default=os.path.join(PROJECT_ROOT, "sample_data/query_samples.csv"),
        help="Path to query samples CSV"
    )
    parser.add_argument(
        "--api-key", default=None, required=False,
        help="API key for the LLM endpoint"
    )
    parser.add_argument(
        "--api-base", default=None, required=False,
        help="Base URL for the OpenAI-compatible LLM endpoint"
    )
    parser.add_argument(
        "--model", default="gpt-3.5-turbo",
        help="Model name (default: gpt-3.5-turbo)"
    )
    args = parser.parse_args()

    # ── Load data ──
    print("[1/5] Loading table metadata...")
    tables = load_table_metadata(args.metadata_dir)
    print(f"  → Loaded {len(tables)} tables")

    print("[2/5] Loading query samples...")
    samples = load_query_samples(args.query_csv)
    print(f"  → Loaded {len(samples)} samples")

    # ── Compute statistics ──
    print("[3/5] Computing usage statistics...")
    stats = compute_usage_stats(samples, tables)
    for tbl, s in stats.items():
        print(f"  {tbl}: usage={s['usage_count']}, co_tables={list(s['co_tables'].keys())}")

    # ── Initialize LLM client ──
    client = LLMClient(
        api_key=args.api_key or "",
        base_url=args.api_base or "",
        model=args.model,
        dry_run=args.dry_run,
    )

    # ── Run Prompt 1 ──
    print("[4/5] Running Prompt 1: table → keywords ...")
    if args.batch_prompt1:
        keywords_map = run_prompt1_batched(client, tables, stats)
    else:
        keywords_map = run_prompt1(client, tables, stats)
    print(f"  → Got keywords for {len(keywords_map)} tables")

    # ── Run Prompt 2 ──
    print("[5/5] Running Prompt 2: keyword combinations → tables ...")
    inverted_map = run_prompt2(client, tables, stats)
    print(f"  → Got {len(inverted_map)} keyword-combination mappings")

    # ── Save results ──
    os.makedirs(args.output_dir, exist_ok=True)
    save_output(keywords_map, "table_keywords.json", args.output_dir)
    save_output(inverted_map, "keyword_to_tables.json", args.output_dir)
    save_output(stats, "co_tables.json", args.output_dir)

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tables indexed:     {len(keywords_map)}")
    print(f"Keyword mappings:   {len(inverted_map)}")
    for k, v in inverted_map.items():
        print(f"  {k} → {v}")
    print(f"\nOutput directory: {args.output_dir}")
    if args.dry_run:
        print("[NOTE] --dry-run was used; output files contain empty JSON.")


if __name__ == "__main__":
    main()
