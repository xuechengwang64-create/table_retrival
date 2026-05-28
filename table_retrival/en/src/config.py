"""English prompts and configuration for the table retrieval governance pipeline."""

# ── Prompt 1: Table → Keywords (forward index) ──
PROMPT_TABLE_TO_KEYWORDS = """You are a database semantic analysis expert. Based on the metadata and query statistics for each table below, generate a list of Chinese keywords for retrieval purposes.

Core requirements:
1. Keywords MUST be strictly [business nouns / entity nouns], e.g. "网络设备" (network device), "交换机" (switch), "闪存存储设备" (flash storage device), "LUN组" (LUN group), "单板" (slot/board). Absolutely NO attribute or metric words such as "CPU利用率" (CPU utilization), "内存使用率" (memory usage), "带宽" (bandwidth).
2. Keywords MUST originate from vocabulary appearing in the table's query_contexts. Do NOT invent words. Use the English table name (e.g. NetworkDevice → network device → 网络设备) as a semantic bridge, but ensure the final Chinese keywords actually exist in the query set.
3. Leverage the table's business_name_cn and description_cn to extract the business entity concepts it represents.
4. Generate 3-8 keywords per table, covering the primary business scenarios in which this table is queried.
5. List synonyms or near-equivalent terms if they both appear in the query set.

Input data:
## Metadata
Table name: {table_name}
Business name: {business_name_cn}
Description: {description_cn}
High-frequency columns: {columns_info}

## Query Statistics
Usage count: {usage_count}
Query contexts: {query_contexts}
Co-occurring tables (with frequency): {co_tables}

Output strictly as JSON, no comments, no extra text:
{{
  "TableName1": ["keyword1", "keyword2", ...],
  "TableName2": ["keyword1", "keyword2", ...],
  ...
}}"""


# ── Prompt 2: Keyword Combinations → Table Sets (inverted index) ──
PROMPT_KEYWORDS_TO_TABLES = """You are a database query routing expert. Based on all table metadata and query statistics, generate precise mappings from keyword combinations to table sets for recall.

Core rules:
1. [Left side — Keyword constraints]
   - Keywords must be business nouns / entity nouns. NO attribute or metric words.
   - Keywords must originate from vocabulary actually appearing in query_contexts.
   - Each combination uses 2-3 nouns to anchor an independent business scenario.

2. [Right side — Table set constraints]
   - No more than 5 tables.
   - Must include ALL tables most closely associated with this business scenario.
   - High-frequency co-occurring tables MUST be recalled together.

3. [Merge principle]
   - If the table set of keyword combination A fully contains the table set of combination B (B ⊂ A), keep only A. Do NOT keep B separately.
   - Entries that cannot be merged must remain independent.
   - Different business scenarios should be kept separate even if their table sets partially overlap.

4. [Scenario-driven]
   - Each mapping corresponds to one independent Q&A business scenario.
   - Make each combination's coverage reasonably broad, but do NOT cram all tables into a single mapping.

Input data:
## All Table Metadata
{all_tables_metadata}

## All Table Query Statistics
{all_tables_stats}

Output strictly as JSON with keyword arrays as keys and table name arrays as values:
{{
  ["网络设备", "交换机"]: ["NetworkDevice", "NetworkDeviceKPI", "NetworkDeviceSummary"],
  ["单板"]: ["NetworkDevice", "DeviceSlot", "DeviceSlotKPI", "SlotSummary"],
  ["闪存存储设备", "LUN组"]: ["FlashStorage", "LUNGroup"]
}}"""


# ── LLM Configuration ──
import os

LLM_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.3
