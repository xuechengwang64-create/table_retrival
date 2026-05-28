#!/usr/bin/env python3
"""
Table Retrieval — Runtime (English version).

Loads governance-output JSONs and retrieves tables for queries.
No LLM calls — pure keyword-anchored matching.

Usage:
  python runtime.py "Query all switches"
  python runtime.py --explain "Query all alarms for network devices"
  python runtime.py --batch queries.txt
  python runtime.py --interactive
"""

import argparse
import json
import os
import sys

EN_SRC = os.path.dirname(os.path.abspath(__file__))
EN_ROOT = os.path.dirname(EN_SRC)
PROJECT_ROOT = os.path.dirname(EN_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from retriever import TableRetriever


def main():
    parser = argparse.ArgumentParser(description="Table Retrieval Runtime (EN)")
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--explain", "-e", action="store_true")
    parser.add_argument("--batch", "-b", default=None)
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--top-k", "-k", type=int, default=10)
    parser.add_argument("--no-expand", action="store_true")
    parser.add_argument("--table-keywords", default=os.path.join(EN_ROOT, "output/table_keywords.json"))
    parser.add_argument("--keyword-to-tables", default=os.path.join(EN_ROOT, "output/keyword_to_tables.json"))
    parser.add_argument("--co-tables", default=os.path.join(EN_ROOT, "output/co_tables.json"))
    args = parser.parse_args()

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

    if args.interactive:
        run_interactive(retriever, args.top_k, not args.no_expand)
    elif args.batch:
        run_batch(retriever, args.batch, args.top_k, not args.no_expand)
    elif args.query:
        if args.explain:
            print(retriever.explain(args.query))
        else:
            tables = retriever.retrieve(args.query, top_k=args.top_k, expand_co_tables=not args.no_expand)
            print(f"Query: {args.query}")
            print(f"Tables ({len(tables)}): {json.dumps(tables, ensure_ascii=False)}")
    else:
        parser.print_help()


def run_batch(retriever, filepath, top_k, expand):
    with open(filepath, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
    for q in queries:
        tables = retriever.retrieve(q, top_k=top_k, expand_co_tables=expand)
        print(f"{q}\t→\t{json.dumps(tables, ensure_ascii=False)}")


def run_interactive(retriever, top_k, expand):
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
