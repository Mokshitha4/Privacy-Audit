"""Member / non-member text loading for the `member_nonmember_split` data role."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .schema import build_schema


@dataclass
class MemberSplit:
    member_texts: List[str]
    nonmember_texts: List[str]


def load_member_split(data_cfg: dict) -> MemberSplit:
    """Load member (training) and non-member (held-out) texts per the job config's `data` block.

    `data.path` is a local directory containing the member/non-member files named by
    `data.schema`'s `member_file`/`nonmember_file`. Nothing under `data.path` is ever uploaded.
    """
    schema_cfg = data_cfg.get("schema")
    if not schema_cfg:
        raise ValueError("data.schema is required (describes how to read your member/non-member files).")
    schema = build_schema(schema_cfg)

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
