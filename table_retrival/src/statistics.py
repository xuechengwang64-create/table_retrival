"""Compute table/column usage statistics from query samples."""

import re
from collections import defaultdict

from data_loader import QuerySample, TableMetadata


def extract_table_names_from_sql(sql: str, known_tables: set[str]) -> set[str]:
    """Extract table names referenced in a SQL statement by matching against known table names."""
    found = set()
    # Case-insensitive matching against known table names
    sql_upper = sql.upper()
    for table in known_tables:
        # Use word-boundary-style matching
        if re.search(rf"\b{re.escape(table)}\b", sql, re.IGNORECASE):
            found.add(table)
    return found


def compute_usage_stats(
    samples: list[QuerySample], tables: dict[str, TableMetadata]
) -> dict[str, dict]:
    """
    Compute per-table usage statistics.

    Returns dict keyed by table_name:
    {
        "usage_count": int,
        "query_contexts": list[str],
        "co_tables": dict[str, int],   # co-table name -> co-occurrence count
    }
    """
    known_tables = set(tables.keys())
    # Per-table accumulators
    usage_count: dict[str, int] = defaultdict(int)
    query_contexts: dict[str, list[str]] = defaultdict(list)
    co_table_pairs: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sample in samples:
        sql_tables = extract_table_names_from_sql(sample.sql, known_tables)
        if not sql_tables:
            continue
        for tbl in sql_tables:
            usage_count[tbl] += 1
            query_contexts[tbl].append(sample.query)
            # Count co-tables
            for other in sql_tables:
                if other != tbl:
                    co_table_pairs[tbl][other] += 1

    # Build result
    result = {}
    for tbl in known_tables:
        co_sorted = sorted(co_table_pairs[tbl].items(), key=lambda x: -x[1])
        result[tbl] = {
            "usage_count": usage_count.get(tbl, 0),
            "query_contexts": query_contexts.get(tbl, []),
            "co_tables": dict(co_sorted),
        }
    return result


def format_table_stats_for_prompt1(stats: dict, table_name: str) -> str:
    """Format stats for a single table as prompt1 input."""
    s = stats.get(table_name, {})
    return f"""使用频次: {s.get('usage_count', 0)}
涉及的问题集: {s.get('query_contexts', [])}
共现表及频次: {s.get('co_tables', {})}"""


def format_all_stats_for_prompt2(stats: dict) -> str:
    """Format all table stats as prompt2 input."""
    lines = []
    for tbl, s in stats.items():
        lines.append(f"表名: {tbl}")
        lines.append(f"  使用频次: {s['usage_count']}")
        lines.append(f"  涉及问题: {s['query_contexts']}")
        lines.append(f"  共现表及频次: {s['co_tables']}")
        lines.append("")
    return "\n".join(lines)


def format_all_metadata_for_prompt2(tables: dict[str, TableMetadata]) -> str:
    """Format all table metadata as prompt2 input."""
    lines = []
    for tbl, meta in tables.items():
        lines.append(f"表名: {meta.table_name}")
        lines.append(f"  业务名称: {meta.business_name_cn}")
        lines.append(f"  描述: {meta.description_cn}")
        lines.append(f"  列信息: {meta.columns_info_str}")
        lines.append("")
    return "\n".join(lines)
