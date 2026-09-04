"""Orchestrates one job: validate config -> load model(s) & data -> run each attack -> build report."""
from __future__ import annotations

from . import __version__
from .attacks import REGISTRY
from .data.member_split import load_member_split
from .models.loader import load_model
from .report import build_report, validate_job_config


def run_job(job_cfg: dict) -> dict:
    validate_job_config(job_cfg)

    data_role = job_cfg["data"]["role"]
    if data_role != "member_nonmember_split":
        raise ValueError(
            f"data.role == {data_role!r} is not supported by this build; only 'member_nonmember_split' "
            "is implemented (RAG-specific attacks are a separate follow-up)."
        )

    # Validate every attack's job-level requirements up front, before loading any (potentially
    # very large) model or dataset, so a misconfigured job fails fast and cheaply.
    for attack_cfg in job_cfg["attacks"]:
        REGISTRY[attack_cfg["family"]].validate(job_cfg, attack_cfg)

    model = load_model(job_cfg["model"])
    reference_model = load_model(job_cfg["reference_model"]) if "reference_model" in job_cfg else None
    member_split = load_member_split(job_cfg["data"])

    output_cfg = job_cfg.get("output", {})
    outputs = [
        REGISTRY[attack_cfg["family"]].run(model, member_split, attack_cfg, output_cfg, reference_model=reference_model)
        for attack_cfg in job_cfg["attacks"]
    ]

    return build_report(job_cfg, outputs, attack_code_version=__version__)
