"""Member / non-member text loading for the `member_nonmember_split` data role."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .schema import resolve_schema


@dataclass
class MemberSplit:
    member_texts: List[str]
    nonmember_texts: List[str]


def load_member_split(data_cfg: dict) -> MemberSplit:
    """Load member (training) and non-member (held-out) texts per the job config's `data` block.

    `data.path` is a local directory containing the member/non-member files named by the
    resolved schema (e.g. train.csv/test.csv for MedQA, or data.schema's member_file/
    nonmember_file for a custom dataset). Nothing under `data.path` is ever uploaded.
    """
    known_dataset = data_cfg.get("known_dataset")
    custom_schema_cfg = data_cfg.get("schema")
    schema = resolve_schema(known_dataset, custom_schema_cfg)

    dataset_dir = Path(data_cfg["path"])
    if not dataset_dir.exists():
        raise FileNotFoundError(f"data.path does not exist: {dataset_dir}")

    member_texts = schema.load_texts(dataset_dir, schema.member_file)
    nonmember_texts = schema.load_texts(dataset_dir, schema.nonmember_file)

    if not member_texts:
        raise ValueError(f"No member texts loaded from {dataset_dir / schema.member_file}")
    if not nonmember_texts:
        raise ValueError(f"No non-member texts loaded from {dataset_dir / schema.nonmember_file}")

    return MemberSplit(member_texts=member_texts, nonmember_texts=nonmember_texts)
