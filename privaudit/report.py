"""JSON Schema validation for job configs, and report construction/validation.

The report is the only artifact meant to ever leave the user's machine. Raw generations
are opt-in (`output.return_raw_generations`) and off by default.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List

import jsonschema

from .attacks.base import AttackOutput

_MODEL_SCHEMA = {
    "type": "object",
    "required": ["source", "identifier", "access"],
    "properties": {
        "source": {"enum": ["huggingface", "local_checkpoint", "api_endpoint", "openai"]},
        "identifier": {
            "type": "string",
            "minLength": 1,
            "description": "HF hub id, local checkpoint path, API endpoint URL, or (for source == 'openai') "
                            "the OpenAI model name, e.g. 'gpt-4o-mini'.",
        },
        "access": {"enum": ["white_box", "black_box"]},
        "base_model": {
            "type": "string",
            "minLength": 1,
            "description": "HF hub id or local path for the base model. Only used when source == "
                            "'local_checkpoint' and 'identifier' is a LoRA/PEFT adapter directory; "
                            "overrides the base model recorded in the adapter's own adapter_config.json.",
        },
        "api_key": {
            "type": "string",
            "minLength": 1,
            "description": "source == 'openai' only. Falls back to the OPENAI_API_KEY environment "
                            "variable when omitted -- prefer the env var over putting a key in a config file.",
        },
        "base_url": {
            "type": "string",
            "minLength": 1,
            "description": "source == 'openai' only. Overrides the API base URL (default "
                            "https://api.openai.com/v1) to point at an OpenAI-compatible endpoint instead.",
        },
        "finetuning": {
            "type": "object",
            "properties": {
                "regime": {"enum": ["full_ft", "qlora", "dp_sgd"]},
                "loss": {"enum": ["full", "masked"]},
                "epsilon": {"type": ["number", "null"]},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

JOB_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["model", "data", "attacks"],
    "properties": {
        "model": _MODEL_SCHEMA,
        "reference_model": _MODEL_SCHEMA,  # required only by attacks that need it (e.g. EZ_MIA); checked per-attack
        "data": {
            "type": "object",
            "required": ["role", "path", "schema"],
            "properties": {
                "role": {"enum": ["member_nonmember_split", "rag_corpus"]},
                "schema": {
                    "type": "object",
                    "required": ["member_file", "nonmember_file"],
                    "properties": {
                        "format": {"enum": ["csv", "jsonl"]},
                        "member_file": {"type": "string", "minLength": 1},
                        "nonmember_file": {"type": "string", "minLength": 1},
                        "text_field": {"type": "string", "minLength": 1},
                        "text_template": {"type": "string", "minLength": 1},
                    },
                    "additionalProperties": False,
                },
                "path": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        "attacks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["family", "variant"],
                "properties": {
                    "family": {"enum": ["EM", "MIA", "EZ_MIA"]},
                    "variant": {"type": "string", "minLength": 1},
                    "params": {"type": "object"},
                },
                "additionalProperties": False,
            },
        },
        "output": {
            "type": "object",
            "properties": {"return_raw_generations": {"type": "boolean"}},
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

REPORT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["job_id", "model_id", "attack_code_version", "generated_at", "metrics", "canonical_scores"],
    "properties": {
        "job_id": {"type": "string"},
        "model_id": {"type": "string"},
        "attack_code_version": {"type": "string"},
        "generated_at": {"type": "string"},
        "metrics": {"type": "object"},
        "canonical_scores": {"type": "object"},
        "raw_generations": {"type": "object"},
    },
    "additionalProperties": False,
}


def validate_job_config(job_cfg: dict) -> None:
    """Validate a job config against the schema; raise ValueError with a clear message on failure."""
    try:
        jsonschema.validate(job_cfg, JOB_CONFIG_SCHEMA)
    except jsonschema.ValidationError as e:
        path = "job" + "".join(f"[{p!r}]" for p in e.absolute_path)
        raise ValueError(f"Invalid job config at {path}: {e.message}") from e

    families = [a["family"] for a in job_cfg["attacks"]]
    duplicates = {f for f in families if families.count(f) > 1}
    if duplicates:
        raise ValueError(
            f"Duplicate attack families in job config: {sorted(duplicates)}. "
            "Each attack family (EM, MIA, EZ_MIA) may appear at most once per job in this build."
        )


def validate_report(report: dict) -> None:
    try:
        jsonschema.validate(report, REPORT_SCHEMA)
    except jsonschema.ValidationError as e:
        path = "report" + "".join(f"[{p!r}]" for p in e.absolute_path)
        raise ValueError(f"Invalid report at {path}: {e.message}") from e


def build_report(job_cfg: dict, outputs: List[AttackOutput], attack_code_version: str) -> dict:
    metrics: Dict[str, dict] = {}
    canonical_scores: Dict[str, float] = {}
    raw_generations: Dict[str, list] = {}

    for output in outputs:
        metrics[output.family] = output.metrics
        canonical_scores[output.family] = output.canonical_score
        if output.raw_generations is not None:
            raw_generations[output.family] = output.raw_generations

    report = {
        "job_id": str(uuid.uuid4()),
        "model_id": job_cfg["model"]["identifier"],
        "attack_code_version": attack_code_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "canonical_scores": canonical_scores,
    }
    if raw_generations:
        report["raw_generations"] = raw_generations

    validate_report(report)
    return report
