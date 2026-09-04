"""Known clinical dataset schemas + custom schema support.

Each schema knows how to turn one row of a member/non-member split into a
single training-format text string, and where to find the member and
non-member files on disk. Field-name/file-name defaults are carried over
from the original research scripts (EZ-MIA/datasets.py, MIA/run_mia_*.py,
Exact Memorization/test_exact_memorization.py) so results stay comparable.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


def _as_row_dict(row) -> dict:
    if isinstance(row, pd.Series):
        return row.to_dict()
    if isinstance(row, dict):
        return row
    return {"value": row}


def _format_icd_diagnoses(diagnoses_list) -> str:
    if isinstance(diagnoses_list, float) and pd.isna(diagnoses_list):
        return "No codes available"
    if isinstance(diagnoses_list, str):
        diagnoses_list = diagnoses_list.strip()
        if not diagnoses_list:
            return "No codes available"
        try:
            diagnoses_list = ast.literal_eval(diagnoses_list)
        except Exception:
            try:
                safe_env = {"nan": np.nan, "null": None, "true": True, "false": False}
                diagnoses_list = eval(diagnoses_list, safe_env)  # noqa: S307 - trusted local dataset files only
            except Exception:
                return "No codes available"
    if not isinstance(diagnoses_list, list) or len(diagnoses_list) == 0:
        return "No codes available"
    formatted = [
        f"{item['diagnosis']} corresponds to {item['code']}"
        for item in diagnoses_list
        if isinstance(item, dict) and "diagnosis" in item and "code" in item
    ]
    return ", ".join(formatted) if formatted else "No codes available"


@dataclass(frozen=True)
class DatasetSchema:
    """How to load and textify one known (or custom) dataset."""

    name: str
    file_format: str  # "csv" | "jsonl"
    member_file: str
    nonmember_file: str
    row_to_text: Callable[[dict], str]

    def load_texts(self, dataset_dir: Path, file_name: str) -> List[str]:
        path = dataset_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if self.file_format == "csv":
            df = pd.read_csv(path)
        elif self.file_format == "jsonl":
            df = pd.read_json(path, lines=True)
        else:
            raise ValueError(f"Unsupported file_format: {self.file_format}")
        texts = [self.row_to_text(_as_row_dict(row)) for _, row in df.iterrows()]
        return [t for t in texts if t]


def _medqa_row_to_text(row: dict) -> str:
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    parts = []
    if question:
        parts.append(f"Question: {question}")
    if answer:
        parts.append(f"Answer: {answer}")
    return "\n".join(parts).strip()


def _icd_row_to_text(row: dict) -> str:
    text_modified = str(row.get("text_modified", row.get("text", ""))).strip()
    answer = _format_icd_diagnoses(row.get("diagnoses"))
    if not text_modified:
        return ""
    instruction = (
        "As a medical expert, your task is to carefully analyze the clinical note and "
        "complete the following steps:\n"
        "1. Identify all key medical terms within the clinical note. These terms should "
        "include: Diagnoses, Symptoms and Relevant conditions\n"
        "2. For each medical term identified, assign the MOST APPROPRIATE ICD code, "
        "ensuring accuracy and specificity in your choices.\n"
        "3. The clinical note may contain multiple conditions, and your role is to ensure "
        "that each one is identified and accurately mapped to the correct ICD code.\n"
        "Your response should follow this structured output format: [medical term] "
        "corresponds to [ICD code].\nIf no ICD code is found for a term: [term] does not "
        "have a matching ICD code."
    )
    return f"{instruction}\n\nClinical Note:\n{text_modified}\n\n ### Condition: {answer}"


def _mimic_task_row_to_text(row: dict) -> str:
    input_text = str(row.get("input", "")).strip()
    output_text = str(row.get("output", "")).strip()
    if not input_text:
        return output_text
    if not output_text:
        return input_text
    return f"{input_text}\n\n{output_text}"


KNOWN_SCHEMAS: Dict[str, DatasetSchema] = {
    "MedQA": DatasetSchema("MedQA", "csv", "train.csv", "test.csv", _medqa_row_to_text),
    "ICD": DatasetSchema("ICD", "csv", "mimic_train.csv", "mimic_test.csv", _icd_row_to_text),
    "mortality": DatasetSchema("mortality", "jsonl", "mimic4_mortality_train.jsonl", "mimic4_mortality_test.jsonl", _mimic_task_row_to_text),
    "readmission": DatasetSchema("readmission", "jsonl", "mimic4_readmission_train.jsonl", "mimic4_readmission_test.jsonl", _mimic_task_row_to_text),
}


def build_custom_schema(schema_cfg: dict) -> DatasetSchema:
    """Build a DatasetSchema from a job config's `data.schema` block.

    Expected shape:
        {
          "format": "csv" | "jsonl",
          "member_file": "train.csv",
          "nonmember_file": "test.csv",
          "text_field": "question",          # used when text_template is absent
          "text_template": "Q: {question}\\nA: {answer}"   # optional, takes precedence
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
        raise ValueError("Custom data.schema requires either 'text_field' or 'text_template'.")

    return DatasetSchema("custom", file_format, member_file, nonmember_file, row_to_text)


def resolve_schema(known_dataset: Optional[str], custom_schema_cfg: Optional[dict]) -> DatasetSchema:
    if known_dataset and known_dataset != "custom":
        if known_dataset not in KNOWN_SCHEMAS:
            raise ValueError(f"Unknown known_dataset '{known_dataset}'. Expected one of {list(KNOWN_SCHEMAS)} or 'custom'.")
        return KNOWN_SCHEMAS[known_dataset]
    if not custom_schema_cfg:
        raise ValueError("data.known_dataset is 'custom' (or unset) but no data.schema was provided.")
    return build_custom_schema(custom_schema_cfg)
