"""Orchestrates one job: validate config -> load model(s) & data -> run each attack -> build report."""
from __future__ import annotations

import gc
from typing import Callable, Optional

from . import __version__
from .attacks import REGISTRY
from .data.member_split import load_member_split
from .models.loader import LoadedModel, load_model
from .report import build_report, validate_job_config

ProgressCallback = Callable[[str], None]


def _release(loaded_model: LoadedModel | None) -> None:
    """Drop a loaded model's weights/tokenizer so they're eligible for garbage collection.

    A short-lived `privaudit run` process would reclaim this on exit regardless, but a
    long-lived caller that runs multiple jobs in the same process (the UI server) does not --
    without this, each Run accumulates another full model in memory on top of the last one,
    eventually exhausting memory (surfaces as a paging-file/OOM error on Windows).
    """
    if loaded_model is None:
        return
    loaded_model.model = None
    loaded_model.tokenizer = None


def run_job(job_cfg: dict, on_progress: Optional[ProgressCallback] = None) -> dict:
    """Run one job. `on_progress`, if given, is called synchronously with a short human-readable
    stage description at each major step (loading a model, running an attack, ...) -- purely
    informational, optional, and never affects the result. The CLI prints these; the UI streams
    them into a live status area."""
    def _report_progress(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

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

    model = None
    reference_model = None
    try:
        _report_progress(f"Loading model ({job_cfg['model']['identifier']})...")
        model = load_model(job_cfg["model"])

        if "reference_model" in job_cfg:
            _report_progress(f"Loading reference model ({job_cfg['reference_model']['identifier']})...")
            reference_model = load_model(job_cfg["reference_model"])

        _report_progress("Loading dataset...")
        member_split = load_member_split(job_cfg["data"])
        _report_progress(
            f"Loaded {len(member_split.member_texts)} member / "
            f"{len(member_split.nonmember_texts)} non-member sequences."
        )

        output_cfg = job_cfg.get("output", {})
        outputs = []
        for attack_cfg in job_cfg["attacks"]:
            _report_progress(f"Running {attack_cfg['family']}...")
            outputs.append(
                REGISTRY[attack_cfg["family"]].run(model, member_split, attack_cfg, output_cfg, reference_model=reference_model)
            )

        _report_progress("Building report...")
        return build_report(job_cfg, outputs, attack_code_version=__version__)
    finally:
        _release(model)
        _release(reference_model)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
