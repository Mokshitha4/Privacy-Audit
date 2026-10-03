"""Dataset schema: how to turn one row of a member/non-member split into training-format text.

Every dataset is described the same way via the job config's `data.schema` block -- there is
no built-in special-casing for any particular dataset (MedQA, ICD, etc.); you always tell the
package which files to read and how to build text from a row.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import pandas as pd


def _as_row_dict(row) -> dict:
    if isinstance(row, pd.Series):
        return row.to_dict()
    if isinstance(row, dict):
        return row
    return {"value": row}


@dataclass(frozen=True)
class DatasetSchema:
    """How to load and textify a member/non-member dataset."""

    file_format: str  # "csv" | "jsonl"
    member_file: str
    nonmember_file: str
    row_to_text: Callable[[dict], str]

    def load_texts(self, dataset_dir: Path, file_name: str) -> List[str]:
        path = dataset_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        looks_like_jsonl = path.suffix.lower() in (".jsonl", ".json")
        if self.file_format == "csv":
            try:
                df = pd.read_csv(path)
            except pd.errors.ParserError as e:
                hint = (
                    f" {path.name} looks like JSONL, not CSV, from its extension; "
                    "set data.schema.format to \"jsonl\" if so."
                ) if looks_like_jsonl else ""
                raise ValueError(f"Failed to parse {path} as CSV (data.schema.format == 'csv'): {e}{hint}") from e
        elif self.file_format == "jsonl":
            try:
                df = pd.read_json(path, lines=True)
            except ValueError as e:
                hint = "" if looks_like_jsonl else (
                    f" {path.name} doesn't look like JSONL from its extension; "
                    "set data.schema.format to \"csv\" if it's actually CSV."
                )
                raise ValueError(f"Failed to parse {path} as JSONL (data.schema.format == 'jsonl'): {e}{hint}") from e
        else:
            raise ValueError(f"Unsupported file_format: {self.file_format}")
        texts = [self.row_to_text(_as_row_dict(row)) for _, row in df.iterrows()]
        return [t for t in texts if t]


def build_schema(schema_cfg: dict) -> DatasetSchema:
    """Build a DatasetSchema from a job config's `data.schema` block.

    Expected shape:
        {
          "format": "csv" | "jsonl",
          "member_file": "train.csv",
          "nonmember_file": "test.csv",
          "text_field": "question",          # used when text_template is absent
          "text_template": "Q: {question}\\nA: {answer}"   # optional, takes precedence;
                                                             # supports nested access, e.g.
                                                             # "{dialog[0][content]}"
        }
    """
    file_format = schema_cfg.get("format", "csv")
    member_file = schema_cfg["member_file"]
    nonmember_file = schema_cfg["nonmember_file"]
    text_template = schema_cfg.get("text_template")
    text_field = schema_cfg.get("text_field")

    if text_template:
        def row_to_text(row: dict, _template=text_template) -> str:
            try:
                return _template.format(**row)
            except KeyError:
                return ""
    elif text_field:
        def row_to_text(row: dict, _field=text_field) -> str:
            return str(row.get(_field, "")).strip()
    else:
        raise ValueError("data.schema requires either 'text_field' or 'text_template'.")

    return DatasetSchema(file_format, member_file, nonmember_file, row_to_text)
