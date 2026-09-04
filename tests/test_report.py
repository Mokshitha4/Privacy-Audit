from __future__ import annotations

from privaudit.attacks.base import AttackOutput
from privaudit.report import build_report, validate_report


def test_build_report_shape_and_canonical_scores():
    job_cfg = {
        "model": {"source": "huggingface", "identifier": "gpt2", "access": "white_box"},
        "data": {"role": "member_nonmember_split", "known_dataset": "MedQA", "path": "/data"},
        "attacks": [{"family": "EM", "variant": "default"}, {"family": "MIA", "variant": "default"}],
    }
    outputs = [
        AttackOutput(family="EM", metrics={"mem_at_50": 0.1}, canonical_score=0.1),
        AttackOutput(family="MIA", metrics={"tpr_at_5pct_fpr": 0.05}, canonical_score=0.05),
    ]
    report = build_report(job_cfg, outputs, attack_code_version="0.1.0")

    assert report["model_id"] == "gpt2"
    assert report["attack_code_version"] == "0.1.0"
    assert report["metrics"] == {"EM": {"mem_at_50": 0.1}, "MIA": {"tpr_at_5pct_fpr": 0.05}}
    assert report["canonical_scores"] == {"EM": 0.1, "MIA": 0.05}
    assert "raw_generations" not in report  # not requested, must not leak by default
    validate_report(report)  # re-validate against the output schema


def test_build_report_includes_raw_generations_only_when_present():
    job_cfg = {
        "model": {"source": "huggingface", "identifier": "gpt2", "access": "white_box"},
        "data": {"role": "member_nonmember_split", "path": "/data", "known_dataset": "custom", "schema": {}},
        "attacks": [{"family": "EM", "variant": "default"}],
    }
    outputs = [
        AttackOutput(
            family="EM",
            metrics={"mem_at_50": 0.2},
            canonical_score=0.2,
            raw_generations=[{"prefix": "p", "groundtruth": "g", "generation": "o"}],
        )
    ]
    report = build_report(job_cfg, outputs, attack_code_version="0.1.0")
    assert report["raw_generations"] == {"EM": [{"prefix": "p", "groundtruth": "g", "generation": "o"}]}
