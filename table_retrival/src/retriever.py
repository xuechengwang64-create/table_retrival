"""
Runtime table retriever — no LLM, pure keyword-anchored matching.

Two retrieval strategies:
  Strategy A (JSON 2): keyword-combination → precise table set (primary)
  Strategy B (JSON 1): query keywords → individual table scoring (fallback)
"""

import json
import re
from collections import defaultdict


class TableRetriever:
    """
    Query → tables, using governance-output JSONs.

    Parameters
    ----------
    table_keywords_path : str
        Path to Prompt-1 output: {table_name: [keyword, ...]}
    keyword_to_tables_path : str
        Path to Prompt-2 output: list of {keywords: [...], tables: [...]}
        or dict with stringified-array keys.
    co_tables_path : str, optional
        Path to co-table stats for expansion on JSON-1 fallback hits.
    """

    def __init__(
        self,
        table_keywords_path: str,
        keyword_to_tables_path: str,
        co_tables_path: str | None = None,
    ):
        self.table_keywords: dict[str, list[str]] = self._load_json(table_keywords_path)
        self.keyword_combos: list[dict] = self._load_combo_mappings(keyword_to_tables_path)
        self.co_tables: dict[str, list[str]] = self._load_co_tables(co_tables_path)

        # Pre-compute: flattened set of all keywords for quick lookup
        self._all_keywords: set[str] = set()
        for combo in self.keyword_combos:
            self._all_keywords.update(combo["keywords"])

    # ── loading ──────────────────────────────────────────────

    @staticmethod
    def _load_json(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def _load_combo_mappings(cls, path: str) -> list[dict]:
        """Normalize JSON 2 into list-of-{keywords, tables}, handling both formats."""
        raw = cls._load_json(path)
        if isinstance(raw, list):
            # Already the preferred format
            return raw
        # Dict format: keys could be stringified arrays like '["A","B"]' or "A, B"
        result = []
        for key_str, tables in raw.items():
            keywords = cls._parse_keywords_key(key_str)
            if keywords:
                result.append({"keywords": keywords, "tables": tables})
        return result

    @staticmethod
    def _parse_keywords_key(key_str: str) -> list[str]:
        """Parse a keyword key from various formats into a list."""
        # Try JSON array format: '["A", "B"]'
        try:
            parsed = json.loads(key_str)
            if isinstance(parsed, list):
                return [str(k).strip() for k in parsed if str(k).strip()]
        except (json.JSONDecodeError, TypeError):
            pass
        # Try comma/semicolon separated
        parts = re.split(r"[,;，；]", key_str)
        parts = [p.strip().strip("\"'") for p in parts if p.strip()]
        return parts

    @classmethod
    def _load_co_tables(cls, path: str | None) -> dict[str, list[str]]:
        if path is None:
            return {}
        raw = cls._load_json(path)
        # Normalize: {table: {co_tables: {t1: n, t2: n}}} → {table: [t1, t2]}
        result = {}
        for tbl, info in raw.items():
            if isinstance(info, dict):
                co = info.get("co_tables", {})
                if isinstance(co, dict):
                    result[tbl] = list(co.keys())
                else:
                    result[tbl] = co if isinstance(co, list) else []
            else:
                result[tbl] = []
        return result

    # ── keyword extraction from query ────────────────────────

    def extract_keywords(self, query: str) -> list[str]:
        """Extract known keywords present in the query text (substring match)."""
        found = []
        for kw in self._all_keywords:
            if kw in query:
                found.append(kw)
        # Sort by length descending so longer (more specific) matches come first
        found.sort(key=len, reverse=True)
        return found

    # ── Strategy A: JSON 2 combination match ─────────────────

    def match_combinations(self, query: str, min_match: int = 1) -> list[dict]:
        """
        Find keyword-combination entries with at least `min_match` keywords
        appearing in the query.

        Returns list of {keywords, tables, match_count, match_ratio} sorted by
        match_ratio descending (more specific matches rank higher).
        """
        hits = []
        for combo in self.keyword_combos:
            kws = combo["keywords"]
            matched = [kw for kw in kws if kw in query]
            if len(matched) >= min_match:
                hits.append({
                    "keywords": kws,
                    "tables": combo["tables"],
                    "matched": matched,
                    "match_ratio": len(matched) / len(kws),
                })
        # Prefer higher coverage of the combination (more specific scene match)
        hits.sort(key=lambda h: (-h["match_ratio"], -len(h["matched"])))
        return hits

    # ── Strategy B: JSON 1 per-table keyword scoring ─────────

    def score_tables_by_keywords(self, query: str) -> dict[str, float]:
        """
        Score each table by how many of its keywords appear in the query.
        Returns {table_name: score}.
        """
        scores = {}
        for tbl, kws in self.table_keywords.items():
            hit_count = sum(1 for kw in kws if kw in query)
            if hit_count > 0:
                # Score = hit ratio (normalised against keyword count)
                scores[tbl] = hit_count / len(kws) if kws else 0
        return scores

    # ── Co-table expansion ───────────────────────────────────

    def expand_with_co_tables(self, tables: set[str]) -> set[str]:
        """Add co-occurrent tables for each table in the set."""
        expanded = set(tables)
        for tbl in tables:
            for co in self.co_tables.get(tbl, []):
                expanded.add(co)
        return expanded

    # ── Combined retrieval ───────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        expand_co_tables: bool = True,
    ) -> list[str]:
        """
        Main entry: query → ranked table list.

        Algorithm
        ---------
        1. Strategy A: keyword-combination matches (JSON 2) → Tier-1 recall.
           Collect all keywords consumed by Strategy A.
        2. Strip consumed keywords from query → residual query.
        3. Strategy B: score tables against the residual query only.
           This prevents already-consumed signal from pulling in noise.
        4. Co-table expansion on Tier-2 only.
        """
        result_tables: list[str] = []
        seen: set[str] = set()

        # ── Tier 1: combination-level hits ──
        combo_hits = self.match_combinations(query)
        consumed_kw: set[str] = set()
        for hit in combo_hits:
            consumed_kw.update(hit["matched"])
            for tbl in hit["tables"]:
                if tbl not in seen:
                    result_tables.append(tbl)
                    seen.add(tbl)

        # ── Tier 2: strip consumed keywords, match residual ──
        residual = query
        for kw in sorted(consumed_kw, key=len, reverse=True):
            residual = residual.replace(kw, "")
        scores = self.score_tables_by_keywords(residual)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        tier2_tables = []
        for tbl, score in ranked:
            if tbl not in seen:
                tier2_tables.append(tbl)
                seen.add(tbl)

        # Co-table expansion only on Tier-2
        if expand_co_tables and self.co_tables and tier2_tables:
            expanded = self.expand_with_co_tables(set(tier2_tables))
            for tbl in expanded:
                if tbl not in seen:
                    tier2_tables.append(tbl)
                    seen.add(tbl)

        result_tables.extend(tier2_tables)
        return result_tables[:top_k]

    # ── Diagnostic ───────────────────────────────────────────

    def explain(self, query: str) -> str:
        """Return a human-readable trace of the retrieval decision."""
        lines = [f"Query: {query}", "=" * 50]

        # Replicate retrieve logic inline for trace visibility
        result_tables: list[str] = []
        seen: set[str] = set()

        combo_hits = self.match_combinations(query)
        consumed_kw: set[str] = set()
        if combo_hits:
            lines.append("\n[Strategy A] Combination matches (JSON 2):")
            for h in combo_hits:
                consumed_kw.update(h["matched"])
                lines.append(
                    f"  [{h['match_ratio']:.0%}] matched {h['matched']} "
                    f"in {h['keywords']} → {h['tables']}"
                )
                for tbl in h["tables"]:
                    if tbl not in seen:
                        result_tables.append(tbl)
                        seen.add(tbl)
        else:
            lines.append("\n[Strategy A] No combination matches.")

        # Build residual
        residual = query
        for kw in sorted(consumed_kw, key=len, reverse=True):
            residual = residual.replace(kw, "")
        lines.append(f"\n  → consumed keywords: {sorted(consumed_kw)}")
        lines.append(f"  → residual query:   \"{residual}\"")

        scores = self.score_tables_by_keywords(residual)
        if scores:
            lines.append(f"\n[Strategy B] Keyword scores against residual (JSON 1):")
            for tbl, s in sorted(scores.items(), key=lambda x: -x[1]):
                lines.append(f"  {tbl}: {s:.2f}  (keywords: {self.table_keywords.get(tbl, [])})")
        else:
            lines.append("\n[Strategy B] No keyword matches in residual.")

        tier2 = []
        for tbl, s in sorted(scores.items(), key=lambda x: -x[1]):
            if tbl not in seen:
                tier2.append(tbl)
                seen.add(tbl)
        if self.co_tables and tier2:
            expanded = self.expand_with_co_tables(set(tier2))
            for tbl in expanded:
                if tbl not in seen:
                    tier2.append(tbl)
                    seen.add(tbl)
        result_tables.extend(tier2)

        lines.append(f"\nFinal top-{len(result_tables)}: {result_tables}")
        return "\n".join(lines)
