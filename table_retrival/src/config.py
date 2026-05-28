"""Prompts and configuration for the table retrieval governance pipeline."""

# ── Prompt 1: 表 → 关键词（正排映射） ──
PROMPT_TABLE_TO_KEYWORDS = """你是一个数据库语义分析专家。请基于以下每张表的元数据与查询统计，为每张表生成用于检索的中文关键词列表。

核心要求：
1. 关键词必须严格是【业务名词/实体名词】，例如"网络设备"、"交换机"、"闪存存储设备"、"LUN组"、"单板"。绝对禁止属性词或指标词，如"CPU利用率"、"内存使用率"、"带宽"。
2. 关键词必须来源于该表的 query_contexts 中出现过的词汇，不得凭空创造。可借助英文表名（如 EntNetworkElement → 网络设备）的语义作为桥梁，但要确保最终选出的中文关键词确实在 query 集中存在。
3. 结合表的 business_name_cn 和 description_cn，提取该表代表的业务实体概念。
4. 每张表生成 3-8 个关键词，应覆盖该表被查询的主要业务场景。
5. 同义词或高度等价词如都在 query 集中出现都应列出。

输入数据：
## 元数据
表名: {table_name}
业务名称: {business_name_cn}
描述: {description_cn}
高频列信息: {columns_info}

## 查询统计
使用频次: {usage_count}
涉及的问题集: {query_contexts}
共现表及频次: {co_tables}

输出严格 JSON，无注释、无额外文字：
{{
  "TableName1": ["关键词1", "关键词2", ...],
  "TableName2": ["关键词1", "关键词2", ...],
  ...
}}"""


# ── Prompt 2: 关键词组合 → 表集合（倒排映射） ──
PROMPT_KEYWORDS_TO_TABLES = """你是一个数据库查询路由专家。请基于所有表的元数据和查询统计，生成关键词组合到召回表集合的精确映射。

核心规则：
1. 【左侧·关键词约束】
   - 关键词必须是业务名词/实体名词，禁止属性词或指标词
   - 关键词必须来源于 query_contexts 中实际出现的词汇
   - 每个组合用 2-3 个名词锚定一个独立业务场景

2. 【右侧·表集合约束】
   - 不超过 5 张表
   - 必须包含该业务场景下最紧密关联的所有表
   - 高频共现表必须一并召回

3. 【合并原则】
   - 若关键词组合 A 的召回表集合能完全包含组合 B 的召回表集合（B ⊂ A），只保留 A，不单独保留 B
   - 不能合并的必须独立成条
   - 不同业务场景即使有部分表重叠也应分别保留

4. 【场景驱动】
   - 每条映射对应一类独立的业务问答场景
   - 适当拉宽每个组合的覆盖面，但不把所有表挤到一个映射里

输入数据：
## 所有表元数据
{all_tables_metadata}

## 所有表查询统计
{all_tables_stats}

输出严格 JSON，键为关键词数组，值为表名数组：
{{
  ["网络设备", "交换机"]: ["EntNetworkElement", "EntNetworkElement2kpi", "kpi_network_device_info"],
  ["单板"]: ["EntNetworkElement", "EnterpriseSlot", "EnterpriseSlot2kpi", "kpi_network_slot_info"],
  ["闪存存储设备", "LUN组"]: ["StorageDevice", "LUNGroup"]
}}"""


# ── LLM 配置 ──
import os

# 通用配置
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# OpenAI 兼容接口（设置了 LLM_API_BASE 即启用，否则走 Anthropic 原生 API）
LLM_API_BASE = os.getenv("LLM_API_BASE", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
