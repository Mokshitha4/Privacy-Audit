"""Shared contract every attack module implements.

Each attack module (em.py, mia.py, ez_mia.py) exposes:

    FAMILY: str
    validate(job_cfg: dict, attack_cfg: dict) -> None
        Raise ValueError with a clear message if required inputs are missing
        (e.g. a PA-style attack would require a data.schema with specific fields;
        EM does not require anything beyond the member split).
    run(model, member_split, attack_cfg, output_cfg, reference_model=None) -> AttackOutput

`run` never writes anything to disk or the network itself -- it returns an
AttackOutput and the caller (runner.py) decides what to keep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AttackOutput:
    family: str
    metrics: Dict[str, Any]
    canonical_score: float
    raw_generations: Optional[List[dict]] = field(default=None)
