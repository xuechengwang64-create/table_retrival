"""Load metadata YAML files and query samples CSV."""

import csv
import os
import yaml
from dataclasses import dataclass, field


@dataclass
class ColumnInfo:
    name: str
    business_name: str
    type: str


@dataclass
class TableMetadata:
    table_name: str
    business_name_cn: str
    description_cn: str
    columns: list[ColumnInfo] = field(default_factory=list)

    @property
    def columns_info_str(self) -> str:
        """Format columns as a compact string for the prompt."""
        lines = []
        for c in self.columns:
            lines.append(f"  {c.name} ({c.business_name}, {c.type})")
        return "\n".join(lines)


@dataclass
class QuerySample:
    id: str
    query: str
    sql: str


def load_table_metadata(metadata_dir: str) -> dict[str, TableMetadata]:
    """Load all YAML metadata files from a directory, keyed by table name."""
    tables = {}
    for filename in sorted(os.listdir(metadata_dir)):
        if not filename.endswith((".yaml", ".yml")):
            continue
        filepath = os.path.join(metadata_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        columns = [
            ColumnInfo(name=c["name"], business_name=c.get("business_name", ""), type=c.get("type", ""))
            for c in raw.get("columns", [])
        ]
        meta = TableMetadata(
            table_name=raw["table_name"],
            business_name_cn=raw.get("business_name_cn", ""),
            description_cn=raw.get("description_cn", ""),
            columns=columns,
        )
        tables[meta.table_name] = meta
    return tables


def load_query_samples(csv_path: str) -> list[QuerySample]:
    """Load query samples from a CSV file (id, query, sql)."""
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(QuerySample(id=row["id"], query=row["query"], sql=row["sql"]))
    return samples
