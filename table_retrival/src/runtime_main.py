#!/usr/bin/env python3
"""
Table Retrieval — Runtime Pipeline Demo.

Loads governance-output JSONs and retrieves tables for queries.
No LLM calls — pure keyword-anchored matching.

Usage:
  python runtime_main.py "查询全部的交换机"
  python runtime_main.py --explain "查询交换机的CPU利用率"
  python runtime_main.py --batch queries.txt
  python runtime_main.py --interactive
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from retriever import TableRetriever


def main():
    parser = argparse.ArgumentParser(description="Table Retrieval — Runtime")
    parser.add_argument(
        "query", nargs="?", default=None,
        help="Natural language query string"
    )
    parser.add_argument(
        "--explain", "-e", action="store_true",
        help="Show matching trace for the query"
    )
    parser.add_argument(
        "--batch", "-b", default=None,
        help="Path to file with one query per line"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Interactive REPL mode"
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=10,
        help="Max tables to return (default: 10)"
    )
    parser.add_argument(
        "--no-expand", action="store_true",
        help="Disable co-table expansion"
    )
    parser.add_argument(
        "--table-keywords", default=os.path.join(PROJECT_ROOT, "output/table_keywords.json"),
        help="Path to table_keywords.json (Prompt 1 output)"
    )
    parser.add_argument(
        "--keyword-to-tables", default=os.path.join(PROJECT_ROOT, "output/keyword_to_tables.json"),
        help="Path to keyword_to_tables.json (Prompt 2 output)"
    )
    parser.add_argument(
        "--co-tables", default=os.path.join(PROJECT_ROOT, "output/co_tables.json"),
        help="Path to co_tables.json for expansion"
    )
    args = parser.parse_args()

    # ── Load retriever ──
    co_tables_path = None if args.no_expand else (
        args.co_tables if os.path.exists(args.co_tables) else None
    )
    retriever = TableRetriever(
        table_keywords_path=args.table_keywords,
        keyword_to_tables_path=args.keyword_to_tables,
        co_tables_path=co_tables_path,
    )
    print(f"[Loaded] {len(retriever.table_keywords)} tables, "
          f"{len(retriever.keyword_combos)} keyword combinations"
          + (f", {len(retriever.co_tables)} co-table entries" if retriever.co_tables else ""))

    # ── Dispatch ──
    if args.interactive:
        run_interactive(retriever, args.top_k, not args.no_expand)
    elif args.batch:
        run_batch(retriever, args.batch, args.top_k, not args.no_expand)
    elif args.query:
        run_single(retriever, args.query, args.top_k, args.explain, not args.no_expand)
    else:
        parser.print_help()


def run_single(retriever: TableRetriever, query: str, top_k: int, explain: bool, expand: bool):
    if explain:
        print(retriever.explain(query))
    else:
        tables = retriever.retrieve(query, top_k=top_k, expand_co_tables=expand)
        print(f"Query: {query}")
        print(f"Tables ({len(tables)}): {json.dumps(tables, ensure_ascii=False)}")


def run_batch(retriever: TableRetriever, filepath: str, top_k: int, expand: bool):
    with open(filepath, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
    for q in queries:
        tables = retriever.retrieve(q, top_k=top_k, expand_co_tables=expand)
        print(f"{q}\t→\t{json.dumps(tables, ensure_ascii=False)}")


def run_interactive(retriever: TableRetriever, top_k: int, expand: bool):
    print("Enter queries (type 'explain <query>' for trace, 'quit' to exit):")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() == "quit":
            break
        if line.lower().startswith("explain "):
            print(retriever.explain(line[8:]))
        else:
            tables = retriever.retrieve(line, top_k=top_k, expand_co_tables=expand)
            print(f"→ {json.dumps(tables, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
